import argparse
import asyncio


def main():
    parser = argparse.ArgumentParser(
        prog="andrewcli",
        description="AndrewCLI - A lightweight python agent",
    )
    # --tray and --server are mutually exclusive "mode" selectors. --voice
    # is intentionally NOT in this group: it augments whichever mode is
    # active. ``--voice`` alone adds voice I/O to the default CLI;
    # ``--tray --voice`` adds it to the tray.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--tray",
        action="store_true",
        help="launch the system tray GUI in background",
    )
    mode.add_argument(
        "--server",
        action="store_true",
        help="launch the FastAPI server",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="add wake-word STT + streaming TTS to the CLI or tray mode",
    )
    parser.add_argument("--host", default="0.0.0.0", help="server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="server port (default: 8000)")
    args = parser.parse_args()

    if args.voice:
        # Voice pipelines log through stdlib logging; configure it once.
        # Do this even in tray mode: the tray subprocess redirects stdout
        # to ~/.andrewcli/tray.log so voice diagnostics land there.
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        for noisy in ("httpx", "httpcore", "urllib3", "hpack",
                      "filelock", "huggingface_hub", "faster_whisper"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.tray:
        import os
        import subprocess
        import sys
        log_path = os.path.expanduser("~/.andrewcli/tray.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log = open(log_path, "a")
        cmd = [sys.executable, "-m", "src.tray"]
        if args.voice:
            cmd.append("--voice")
        subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=log,
            stderr=log,
        )
        print(f"Andrew tray started in background. Logs: {log_path}")

    elif args.server:
        import uvicorn
        uvicorn.run("src.server:app", host=args.host, port=args.port)

    else:
        from src.app import AndrewCLI
        try:
            andrew = AndrewCLI(voice_enabled=args.voice)
            asyncio.run(andrew.run())
        except KeyboardInterrupt:
            print("\nAndrew stopped. Goodbye!")


if __name__ == "__main__":
    main()
