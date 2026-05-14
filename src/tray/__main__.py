import argparse

from src.tray.app import main

parser = argparse.ArgumentParser(prog="python -m src.tray")
parser.add_argument("--host", default="0.0.0.0", help="API server host")
parser.add_argument("--port", type=int, default=8000, help="API server port")
args = parser.parse_args()

from src.shared.config import Config
if Config().server_enabled:
    from src.core.server import server
    server.start_background(args.host, args.port)

main()
