import argparse
import asyncio


def main():
    # Importing paths triggers the first-run seeding of
    # ~/.config/andrewcli/ from the bundled defaults and adds the
    # config dir to sys.path, so subsequent ``import domains.*`` and
    # ``import events.*`` calls resolve against the user's runtime
    # tree. Done first so that even subprocess-spawning modes (--tray)
    # have a fully provisioned config before forking.
    import src.shared.paths  # noqa: F401

    parser = argparse.ArgumentParser(
        prog="andrewcli",
        description="AndrewCLI - A lightweight python agent",
    )
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
    parser.add_argument("--host", default="0.0.0.0", help="server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="server port (default: 8000)")
    args = parser.parse_args()

    if args.tray:
        import os
        import subprocess
        import sys
        # Anchor the tray to the directory the user launched from. The
        # env var propagates through Popen so the child's
        # ``src.shared.paths`` resolves the same LAUNCH_DIR even though
        # the subprocess has its own Python interpreter.
        launch_dir = os.path.abspath(os.getcwd())
        log_path = os.path.join(launch_dir, ".andrewcli", "tray.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log = open(log_path, "a")
        cmd = [sys.executable, "-m", "src.tray"]
        cmd += ["--host", args.host, "--port", str(args.port)]
        env = {**os.environ, "ANDREW_LAUNCH_DIR": launch_dir}
        subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=log,
            stderr=log,
            cwd=launch_dir,
            env=env,
        )
        print(f"Andrew tray started in background. Logs: {log_path}")
        # Mirror the gating done in the tray child process (src/tray/__main__.py)
        # so the parent's user-facing message doesn't lie about what's running.
        from src.shared.config import Config
        if Config().server_enabled:
            print(f"API server will be available at http://{args.host}:{args.port}")

    elif args.server:
        import uvicorn
        uvicorn.run("src.core.server:app", host=args.host, port=args.port)

    else:
        from src.cli.app import AndrewCLI
        from src.shared.config import Config
        if Config().server_enabled:
            from src.core.server import server
            server.start_background(args.host, args.port)
            print(f"API server running on http://{args.host}:{args.port}")
        try:
            andrew = AndrewCLI()
            asyncio.run(andrew.run())
        except KeyboardInterrupt:
            print("\nAndrew stopped. Goodbye!")


if __name__ == "__main__":
    main()
