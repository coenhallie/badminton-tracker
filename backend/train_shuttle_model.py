#!/usr/bin/env python3
"""
Train a custom YOLO model specifically for badminton shuttle detection.
Uses the "Badminton Shuttle Detector" dataset from Roboflow.

Usage:
    cd backend
    source venv/bin/activate
    python train_shuttle_model.py

The trained model will be saved in models/shuttle/weights/best.pt
"""

from ultralytics import YOLO
import os
import argparse

# Configuration
DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets", "Badminton Shuttle Detector.v1i.yolo26")
DATA_YAML = os.path.join(DATASET_PATH, "data.yaml")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")

# Default training parameters optimized for shuttle detection
DEFAULT_EPOCHS = 150  # More epochs for small object detection
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_MODEL = "yolo26n.pt"  # YOLO26 Nano model - NMS-free, optimized for real-time detection


def train_model(
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    base_model: str = DEFAULT_MODEL,
    resume: bool = False,
):
    """
    Train the YOLO model on the shuttle detection dataset.
    
    Args:
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        base_model: Base YOLO model to fine-tune
        resume: Whether to resume from last checkpoint
    """
    print("=" * 60)
    print("SHUTTLE DETECTION MODEL TRAINING")
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
    
    # Train the model with settings optimized for small object detection
    print("\nStarting training...")
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=OUTPUT_DIR,
        name="shuttle",
        exist_ok=True,
        resume=resume,
        # Performance optimizations
        workers=4,
        patience=30,  # Higher patience for small object learning
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        # Data augmentation optimized for shuttlecock detection
        hsv_h=0.015,  # Hue augmentation (shuttlecock is usually white/feathered)
        hsv_s=0.5,    # Moderate saturation augmentation
        hsv_v=0.5,    # Value augmentation (important for white objects)
        degrees=15.0,  # Rotation augmentation
        translate=0.2, # Translation for better position learning
        scale=0.5,    # Scale augmentation for different distances
        flipud=0.0,   # Vertical flip (not useful for badminton)
        fliplr=0.5,   # Horizontal flip
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.1,    # Light mixup augmentation
        # Loss function weights - emphasize box precision for small objects
        box=7.5,      # Box loss weight (increased for small objects)
        cls=0.5,      # Classification loss weight
        dfl=1.5,      # Distribution focal loss
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    # Find the best weights
    best_weights = os.path.join(OUTPUT_DIR, "shuttle", "weights", "best.pt")
    if os.path.exists(best_weights):
        print(f"Best model saved to: {best_weights}")
        print("\nTo use this model for shuttle detection, update your .env file:")
        print(f"  YOLO_SHUTTLE_MODEL={best_weights}")
    
    return results


def validate_model():
    """Validate the trained model on the test set."""
    model_path = os.path.join(OUTPUT_DIR, "shuttle", "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        print("Run training first: python train_shuttle_model.py")
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
    model_path = os.path.join(OUTPUT_DIR, "shuttle", "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        return
    
    print(f"Exporting model to {format} format...")
    model = YOLO(model_path)
    model.export(format=format)
    print(f"Export complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model for shuttle detection")
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
