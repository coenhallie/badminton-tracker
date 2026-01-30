#!/usr/bin/env python3
"""
Train a custom YOLO model for badminton detection (person, racket, shuttle).
Uses the local dataset exported from Roboflow.

Usage:
    cd backend
    source venv/bin/activate
    python train_badminton_model.py

The trained model will be saved in runs/detect/badminton/weights/best.pt
"""

from ultralytics import YOLO
import os
import argparse

# Configuration
DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets", "badminton.v1i.yolo26")
DATA_YAML = os.path.join(DATASET_PATH, "data.yaml")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")

# Default training parameters
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_MODEL = "yolov8n.pt"  # Nano model - fast, good for real-time detection


def train_model(
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    base_model: str = DEFAULT_MODEL,
    resume: bool = False,
):
    """
    Train the YOLO model on the badminton dataset.
    
    Args:
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        base_model: Base YOLO model to fine-tune
        resume: Whether to resume from last checkpoint
    """
    print("=" * 60)
    print("BADMINTON DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Data config: {DATA_YAML}")
    print(f"Base model: {base_model}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print("=" * 60)
    
    # Verify dataset exists
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load base model
    print(f"\nLoading base model: {base_model}")
    model = YOLO(base_model)
    
    # Train the model
    print("\nStarting training...")
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=OUTPUT_DIR,
        name="badminton",
        exist_ok=True,
        resume=resume,
        # Performance optimizations
        workers=4,
        patience=20,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        # Data augmentation (reasonable defaults for sports)
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,    # Saturation augmentation
        hsv_v=0.4,    # Value augmentation
        degrees=10.0,  # Rotation augmentation
        flipud=0.0,   # Vertical flip (not useful for badminton)
        fliplr=0.5,   # Horizontal flip
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.0,    # Mixup augmentation
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    # Find the best weights
    best_weights = os.path.join(OUTPUT_DIR, "badminton", "weights", "best.pt")
    if os.path.exists(best_weights):
        print(f"Best model saved to: {best_weights}")
        print("\nTo use this model, update your .env file:")
        print(f"  YOLO_BADMINTON_MODEL={best_weights}")
    
    return results


def validate_model():
    """Validate the trained model on the test set."""
    model_path = os.path.join(OUTPUT_DIR, "badminton", "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        print("Run training first: python train_badminton_model.py")
        return
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    print("Running validation on test set...")
    results = model.val(data=DATA_YAML, split="test")
    
    print("\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    
    return results


def export_model(format: str = "onnx"):
    """Export the trained model to different formats."""
    model_path = os.path.join(OUTPUT_DIR, "badminton", "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        return
    
    print(f"Exporting model to {format} format...")
    model = YOLO(model_path)
    model.export(format=format)
    print(f"Export complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model for badminton detection")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Number of epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help=f"Batch size (default: {DEFAULT_BATCH})")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help=f"Image size (default: {DEFAULT_IMGSZ})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Base model (default: {DEFAULT_MODEL})")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    parser.add_argument("--export", type=str, choices=["onnx", "tflite", "coreml"], help="Export model format")
    
    args = parser.parse_args()
    
    if args.validate:
        validate_model()
    elif args.export:
        export_model(args.export)
    else:
        train_model(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            base_model=args.model,
            resume=args.resume,
        )
