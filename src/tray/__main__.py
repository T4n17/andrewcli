import argparse
import logging

from src.tray.app import main

parser = argparse.ArgumentParser(prog="python -m src.tray")
parser.add_argument(
    "--voice",
    action="store_true",
    help="add wake-word STT + streaming TTS to the tray",
)
parser.add_argument("--host", default="0.0.0.0", help="API server host")
parser.add_argument("--port", type=int, default=8000, help="API server port")
args = parser.parse_args()

# The tray runs as a detached subprocess whose stdout/stderr are
# redirected to ~/.andrewcli/tray.log. Configure logging so voice
# diagnostics (and any future subprocess-side INFO messages) land
# there; without this the subprocess runs at WARNING and the log
# file is effectively empty.
if args.voice:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "hpack",
                  "filelock", "huggingface_hub", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

from src.core.server import start_background
start_background(args.host, args.port)

main(voice_enabled=args.voice)
