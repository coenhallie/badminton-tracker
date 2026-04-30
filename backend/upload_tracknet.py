"""
Upload TrackNetV3 checkpoints to Modal Volume.

Prerequisites:
1. Download checkpoints from:
   https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view
2. Extract to get ckpts/TrackNet_best.pt and ckpts/InpaintNet_best.pt

Usage:
    python backend/upload_tracknet.py --tracknet ckpts/TrackNet_best.pt --inpaintnet ckpts/InpaintNet_best.pt
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Upload TrackNetV3 checkpoints to Modal")
    parser.add_argument(
        "--tracknet",
        required=True,
        help="Path to TrackNet_best.pt checkpoint",
    )
    parser.add_argument(
        "--inpaintnet",
        default=None,
        help="Path to InpaintNet_best.pt checkpoint (optional but recommended)",
    )
    args = parser.parse_args()

    try:
        import modal
    except ImportError:
        print("Error: modal package not installed. Run: pip install modal")
        sys.exit(1)

    # Read checkpoint files
    print(f"Reading TrackNet checkpoint: {args.tracknet}")
    with open(args.tracknet, "rb") as f:
        tracknet_data = f.read()
    print(f"  Size: {len(tracknet_data) / 1024 / 1024:.1f} MB")

    inpaintnet_data = None
    if args.inpaintnet:
        print(f"Reading InpaintNet checkpoint: {args.inpaintnet}")
        with open(args.inpaintnet, "rb") as f:
            inpaintnet_data = f.read()
        print(f"  Size: {len(inpaintnet_data) / 1024 / 1024:.1f} MB")

    # Upload to Modal volume directly
    print("\nUploading to Modal volume 'badminton-tracker-models'...")
    vol = modal.Volume.from_name("badminton-tracker-models", create_if_missing=True)

    # Write files to the volume
    with vol.batch_upload() as batch:
        import tempfile, os

        # Write TrackNet checkpoint
        tracknet_tmp = os.path.join(tempfile.gettempdir(), "TrackNet_best.pt")
        with open(tracknet_tmp, "wb") as f:
            f.write(tracknet_data)
        batch.put_file(tracknet_tmp, "/tracknet/TrackNet_best.pt")
        print(f"  Uploaded TrackNet_best.pt ({len(tracknet_data) / 1024 / 1024:.1f} MB)")

        # Write InpaintNet checkpoint
        if inpaintnet_data:
            inpaintnet_tmp = os.path.join(tempfile.gettempdir(), "InpaintNet_best.pt")
            with open(inpaintnet_tmp, "wb") as f:
                f.write(inpaintnet_data)
            batch.put_file(inpaintnet_tmp, "/tracknet/InpaintNet_best.pt")
            print(f"  Uploaded InpaintNet_best.pt ({len(inpaintnet_data) / 1024 / 1024:.1f} MB)")

    print("\nUpload complete! TrackNet is now available for the next video processing run.")
    print("Redeploy the processor: modal deploy backend/modal_supabase_processor.py")


if __name__ == "__main__":
    main()
