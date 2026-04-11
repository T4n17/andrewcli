import argparse
import asyncio


def main():
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
        log_path = os.path.expanduser("~/.andrewcli/tray.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log = open(log_path, "a")
        subprocess.Popen(
            [sys.executable, "-m", "src.tray"],
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
            andrew = AndrewCLI()
            asyncio.run(andrew.run())
        except KeyboardInterrupt:
            print("\nAndrew stopped. Goodbye!")


if __name__ == "__main__":
    main()
