"""Generic "monitor X for changes" event.

A *monitor* periodically runs a user-supplied shell command, snapshots
its output (exit code + stdout + stderr) to a file, and watches that
file for changes via the inherited :class:`FileEvent` mechanism. Any
state that can be observed by a shell command — and in practice that is
nearly anything: web pages (`curl`), HTTP healthchecks (`curl -w`), JSON
fields (`curl | jq`), disk usage (`df`), repo status (`git status`),
processes (`systemctl is-active`), file checksums (`sha256sum`), DB
queries (`psql -tAc`), RSS feeds, custom scripts — fits this single
mechanism.

The shell already provides composition (pipes), authentication
(`curl -H`, `--cacert`), filtering (`grep`, `jq`, `awk`) and retries
(`curl --retry`), so the monitor itself stays small: one class, one
hook, infinite use cases.

Usage
-----
``/monitor "<shell command>" [capture_interval] [poll_interval] [timeout]``

Examples (verbatim slash-command lines)::

    /monitor "curl -sL https://example.com" 300
    /monitor "curl -sL -w '\\n%{http_code}' https://api.example.com/health" 60
    /monitor "df -h /" 600
    /monitor "git -C /path status --porcelain" 30
    /monitor "systemctl is-active nginx" 60
    /monitor "sha256sum /etc/hosts" 120

Design notes
------------
* The snapshot file is rewritten only when the captured bytes' hash
  changes, so the watched path's mtime advances exactly once per real
  change. Idle re-captures of identical output don't cause spurious
  fires.
* The first capture is performed inline before the file watcher starts,
  so the initial population is *not* reported as a change.
* Each capture has a hard timeout so a hung command can't wedge the
  capture loop; on timeout the command is killed and the cycle retries.
* Exit code, stdout, and stderr are all included in the snapshot — a
  command that starts failing is itself a state-change signal, which
  covers healthy↔unhealthy transitions for free.
* The capture loop runs as an independent asyncio task that survives
  every fire and is cancelled together with the event itself.
"""

import asyncio
import hashlib
import os
import subprocess
import tempfile
from typing import Optional

from events.file import FileEvent


class MonitorEvent(FileEvent):
    """Periodically run a shell command and fire when its output changes.

    The shell command is the universal interface: anything whose state
    can be observed from a process can be monitored without writing
    Python. Subclassing is supported but rarely necessary — most use
    cases are a one-line slash command.
    """

    name = "monitor"

    def __init__(
        self,
        command: str = "",
        capture_interval: float = 60.0,
        poll_interval: float = 2.0,
        timeout: float = 30.0,
        snapshot_path: str = "",
        message: str = "",
    ):
        if not command:
            raise ValueError(
                'Usage: /monitor "<shell command>" '
                "[capture_interval] [poll_interval] [timeout]"
            )
        self.command = command
        self.capture_interval = max(1.0, float(capture_interval))
        self.timeout = max(1.0, float(timeout))

        if not snapshot_path:
            slug = hashlib.sha1(command.encode()).hexdigest()[:12]
            snapshot_path = os.path.join(
                tempfile.gettempdir(), f"andrewcli-monitor-{slug}.snap"
            )
        # Ensure the file exists so FileEvent.__init__ can read its mtime
        # without raising; first-capture below will populate it.
        if not os.path.exists(snapshot_path):
            open(snapshot_path, "ab").close()

        super().__init__(
            path=snapshot_path,
            poll_interval=float(poll_interval),
            message=message or "",  # filled in below once self.path is set
        )
        if not message:
            self.message = self._default_message()
        self.description = (
            f"Monitors `{command[:60]}` (capture every {self.capture_interval:g}s)"
        )

        self._last_hash: Optional[str] = self._hash_file()
        self._capture_task: asyncio.Task | None = None
        self._initialised: bool = False

    def _default_message(self) -> str:
        return (
            f"The monitored command output has changed. "
            f"Command: {self.command}\n"
            f"Snapshot file: {self.path}\n"
            f"Read the snapshot file to see the new state."
        )

    # ----- internals -------------------------------------------------------

    def _hash_file(self) -> Optional[str]:
        try:
            with open(self.path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            return None

    def _write_snapshot(self, data: bytes) -> bool:
        """Write *data* to the snapshot file iff its hash changed.

        Returns True if a write happened (and the watched mtime therefore
        advanced), False if the bytes were identical to the last write.
        Writes are atomic via os.replace so partial writes never produce
        spurious mtime changes on the watched path.
        """
        digest = hashlib.sha256(data).hexdigest()
        if digest == self._last_hash:
            return False
        tmp = f"{self.path}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, self.path)
        self._last_hash = digest
        return True

    def _run_sync(self) -> bytes | None:
        """Synchronously run the command; return the framed snapshot bytes.

        Uses :func:`subprocess.run` with a ``timeout`` because the asyncio
        subprocess + ``wait_for`` combination has a known issue on Linux
        where re-awaiting after ``kill()`` blocks until the original
        timer expires. The sync API kills the process *and* drains the
        pipe-reader threads correctly on timeout.

        Returns ``None`` only on a transient spawn failure; a non-zero
        exit code is *not* a failure — it is part of the captured state
        and changes between runs trigger the watcher just like stdout
        changes.
        """
        try:
            result = subprocess.run(
                self.command,
                shell=True,
                capture_output=True,
                timeout=self.timeout,
            )
            rc = result.returncode
            stdout = result.stdout or b""
            stderr = result.stderr or b""
        except subprocess.TimeoutExpired as e:
            rc = -1
            stdout = e.stdout or b""
            stderr = (e.stderr or b"") + (
                f"[monitor] command timed out after {self.timeout:g}s\n".encode()
            )
        except Exception:
            return None

        parts = [f"exit_code: {rc}\n".encode(), b"--stdout--\n", stdout]
        if stderr:
            parts.extend([b"\n--stderr--\n", stderr])
        return b"".join(parts)

    async def _capture(self) -> bytes | None:
        """Run the command in a thread executor and return the framed bytes."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_sync)

    async def _safe_capture(self) -> bytes | None:
        try:
            return await self._capture()
        except asyncio.CancelledError:
            raise
        except Exception:
            return None

    async def _capture_loop(self) -> None:
        """Background task: periodically snapshot the command output to disk."""
        while True:
            await asyncio.sleep(self.capture_interval)
            data = await self._safe_capture()
            if data is not None:
                self._write_snapshot(data)

    # ----- event interface -------------------------------------------------

    async def condition(self):
        if not self._initialised:
            # Perform the first capture inline so the initial population
            # does not count as a change. Reset the FileEvent baseline
            # mtime to the post-capture value, then start the periodic
            # background capture loop.
            data = await self._safe_capture()
            if data is not None:
                self._write_snapshot(data)
            self._last_mtime = self._mtime()
            self._initialised = True
            self._capture_task = asyncio.create_task(
                self._capture_loop(), name="monitor:capture"
            )

        try:
            await super().condition()
        except asyncio.CancelledError:
            if self._capture_task is not None and not self._capture_task.done():
                self._capture_task.cancel()
            raise
