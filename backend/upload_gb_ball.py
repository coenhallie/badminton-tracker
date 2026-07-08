"""
Upload the Good-Badminton yolo11s-ball.pt shuttlecock detector to the Modal
models volume, for the gb_fusion Phase 1 pipeline.

Weight source (Apache-2.0):
https://github.com/yo-WASSUP/Good-Badminton/releases/download/v0.1.0/yolo11s-ball.pt

Usage:
    backend/venv/bin/python backend/upload_gb_ball.py --path /path/to/yolo11s-ball.pt
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Upload GB ball weight to Modal")
    parser.add_argument("--path", required=True, help="Path to yolo11s-ball.pt")
    args = parser.parse_args()

    try:
        import modal
    except ImportError:
        print("Error: modal package not installed. Run: pip install modal")
        sys.exit(1)

    vol = modal.Volume.from_name("badminton-tracker-models", create_if_missing=True)
    with vol.batch_upload(force=True) as batch:
        batch.put_file(args.path, "/gb_ball/yolo11s-ball.pt")
    print("Uploaded to badminton-tracker-models:/gb_ball/yolo11s-ball.pt")


if __name__ == "__main__":
    main()
