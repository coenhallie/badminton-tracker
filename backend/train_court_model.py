#!/usr/bin/env python3
"""
Train a custom YOLO model specifically for badminton court detection.
Uses the "Badminton Court" dataset from Roboflow.

This model detects court regions and lines:
- frontcourt, midcourt-down, midcourt-up, net
- rearcourt-down, rearcourt-up, sideline-left, sideline-right

EPOCH ANALYSIS:
===============
Dataset: ~2,582 training images, 737 validation images, 8 classes
Recommendation: 80-120 epochs with early stopping (patience=40)

For court detection (geometric shapes/lines):
- Court lines are consistent geometric patterns
- Convergence typically occurs between epochs 60-100
- Early stopping with patience=40 will find optimal stopping point
- Default 100 epochs with early stopping is recommended

If training completes without early stopping triggering:
- Model may benefit from more epochs (increase to 150)
- Or the learning rate may need adjustment

Usage:
    cd backend
    source venv/bin/activate
    python train_court_model.py

    # With custom epochs
    python train_court_model.py --epochs 120

The trained model will be saved in models/court/weights/best.pt
"""

from ultralytics import YOLO
import os
import argparse
import json
from datetime import datetime

# Configuration
DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets", "Badminton Court.v1i.yolo26")
DATA_YAML = os.path.join(DATASET_PATH, "data.yaml")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")

# Default training parameters optimized for court detection
# Recommended: 80-120 epochs for court detection with ~2500 training images
DEFAULT_EPOCHS = 100  # Good baseline for court detection
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_MODEL = "yolov8n.pt"  # Nano model - fast, good for real-time detection
DEFAULT_PATIENCE = 40  # Early stopping patience - stops if no improvement for 40 epochs


def analyze_epoch_requirements():
    """
    Analyze and print recommendations for optimal epoch count.
    
    Based on:
    - Dataset size (~2,582 training images)
    - Number of classes (8 geometric court regions)
    - Nature of detection (geometric shapes/lines)
    """
    print("\n" + "=" * 60)
    print("EPOCH ANALYSIS & RECOMMENDATIONS")
    print("=" * 60)
    
    analysis = {
        "dataset_size": "~2,582 training images, 737 validation images",
        "classes": 8,
        "class_names": [
            "frontcourt", "midcourt-down", "midcourt-up", "net",
            "rearcourt-down", "rearcourt-up", "sideline-left", "sideline-right"
        ],
        "detection_type": "Geometric shapes (court lines and regions)",
        "recommended_epochs": "80-120",
        "optimal_starting_epochs": 100,
        "early_stopping_patience": 40,
        "reasoning": [
            "Court detection involves consistent geometric patterns",
            "Medium dataset size (~2.5k images) enables moderate training",
            "8 classes of geometric shapes converge faster than complex objects",
            "Early stopping with patience=40 prevents overfitting automatically",
            "If early stopping triggers before epoch 70, consider reducing epochs",
            "If training completes all epochs, consider increasing to 120-150"
        ]
    }
    
    print(f"\nDataset: {analysis['dataset_size']}")
    print(f"Classes: {analysis['classes']} (geometric court regions)")
    print(f"\nRecommended Epochs: {analysis['recommended_epochs']}")
    print(f"Starting with: {analysis['optimal_starting_epochs']} epochs")
    print(f"Early Stopping Patience: {analysis['early_stopping_patience']}")
    print("\nReasoning:")
    for i, reason in enumerate(analysis['reasoning'], 1):
        print(f"  {i}. {reason}")
    
    print("=" * 60 + "\n")
    
    return analysis


def train_model(
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    base_model: str = DEFAULT_MODEL,
    patience: int = DEFAULT_PATIENCE,
    resume: bool = False,
):
    """
    Train the YOLO model on the court detection dataset.
    
    Args:
        epochs: Number of training epochs (recommended: 80-120)
        imgsz: Image size for training
        batch: Batch size
        base_model: Base YOLO model to fine-tune
        patience: Early stopping patience (recommended: 40)
        resume: Whether to resume from last checkpoint
    """
    # Show epoch analysis
    analyze_epoch_requirements()
    
    print("=" * 60)
    print("BADMINTON COURT DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Data config: {DATA_YAML}")
    print(f"Base model: {base_model}")
    print(f"Epochs: {epochs}")
    print(f"Early Stopping Patience: {patience}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print("=" * 60)
    
    # Verify dataset exists
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"Dataset config not found: {DATA_YAML}\n"
            f"Please ensure the dataset is copied to: {DATASET_PATH}"
        )
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load base model
    print(f"\nLoading base model: {base_model}")
    model = YOLO(base_model)
    
    # Train the model with settings optimized for court detection
    print("\nStarting training...")
    print(f"Training will stop early if no improvement for {patience} epochs")
    print("Monitor the mAP50 metric - court detection typically achieves >0.8 mAP50\n")
    
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=OUTPUT_DIR,
        name="court",
        exist_ok=True,
        resume=resume,
        # Performance optimizations
        workers=4,
        patience=patience,  # Early stopping - critical for finding optimal epochs
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        # Data augmentation optimized for court/line detection
        hsv_h=0.01,   # Minimal hue augmentation (court colors are consistent)
        hsv_s=0.3,    # Light saturation augmentation
        hsv_v=0.4,    # Value augmentation (important for different lighting)
        degrees=5.0,  # Light rotation (courts are typically fixed orientation)
        translate=0.1, # Light translation
        scale=0.3,    # Scale augmentation for different camera angles
        shear=2.0,    # Light shear for perspective variations
        perspective=0.0005,  # Light perspective augmentation
        flipud=0.0,   # No vertical flip (courts have fixed orientation)
        fliplr=0.5,   # Horizontal flip (courts are symmetric)
        mosaic=0.8,   # Moderate mosaic augmentation
        mixup=0.0,    # No mixup for geometric detection
        # Loss function weights for line/region detection
        box=7.5,      # Box loss weight
        cls=0.5,      # Classification loss weight
        dfl=1.5,      # Distribution focal loss
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    # Find the best weights
    best_weights = os.path.join(OUTPUT_DIR, "court", "weights", "best.pt")
    
    # Training summary
    if hasattr(results, 'results_dict'):
        print("\nTraining Summary:")
        print(f"  Epochs completed: {results.epoch}")
        if results.box:
            print(f"  Best mAP50: {results.box.map50:.4f}")
            print(f"  Best mAP50-95: {results.box.map:.4f}")
    
    # Report early stopping status
    results_csv = os.path.join(OUTPUT_DIR, "court", "results.csv")
    if os.path.exists(results_csv):
        import csv
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            actual_epochs = len(rows)
            print(f"\nActual epochs trained: {actual_epochs}")
            if actual_epochs < epochs:
                print(f"  Early stopping triggered at epoch {actual_epochs}")
                print(f"  → This is optimal! No improvement for {patience} epochs")
            else:
                print(f"  Training completed all {epochs} epochs")
                print(f"  → Consider increasing epochs if loss is still decreasing")
    
    if os.path.exists(best_weights):
        print(f"\nBest model saved to: {best_weights}")
        print("\nTo use this model for court detection, update your .env file:")
        print(f"  YOLO_COURT_MODEL={best_weights}")
    
    # Save training metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "epochs_requested": epochs,
        "patience": patience,
        "base_model": base_model,
        "imgsz": imgsz,
        "batch": batch,
        "dataset": DATASET_PATH,
    }
    metadata_path = os.path.join(OUTPUT_DIR, "court", "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return results


def validate_model():
    """Validate the trained model on the test set."""
    model_path = os.path.join(OUTPUT_DIR, "court", "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        print("Run training first: python train_court_model.py")
        return
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    print("Running validation on test set...")
    results = model.val(data=DATA_YAML, split="test")
    
    print("\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    
    # Per-class results
    if hasattr(results.box, 'ap_class_index'):
        class_names = ['frontcourt', 'midcourt-down', 'midcourt-up', 'net',
                       'rearcourt-down', 'rearcourt-up', 'sideline-left', 'sideline-right']
        print("\nPer-class AP50:")
        for i, ap in enumerate(results.box.ap50):
            print(f"  {class_names[i]}: {ap:.4f}")
    
    return results


def export_model(format: str = "onnx"):
    """Export the trained model to different formats."""
    model_path = os.path.join(OUTPUT_DIR, "court", "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        return
    
    print(f"Exporting model to {format} format...")
    model = YOLO(model_path)
    model.export(format=format)
    print(f"Export complete!")


def analyze_only():
    """Just print the epoch analysis without training."""
    analyze_epoch_requirements()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLO model for badminton court detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EPOCH RECOMMENDATIONS:
  - Default: 100 epochs with early stopping (patience=40)
  - Minimum recommended: 80 epochs
  - Maximum recommended: 120 epochs (if no early stopping)
  - Early stopping will automatically find optimal stopping point

EXAMPLES:
  python train_court_model.py                    # Default 100 epochs
  python train_court_model.py --epochs 120      # More epochs if needed
  python train_court_model.py --analyze         # Just show epoch analysis
  python train_court_model.py --validate        # Validate trained model
"""
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, 
                        help=f"Number of epochs (default: {DEFAULT_EPOCHS}, recommended: 80-120)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, 
                        help=f"Batch size (default: {DEFAULT_BATCH})")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, 
                        help=f"Image size (default: {DEFAULT_IMGSZ})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                        help=f"Base model (default: {DEFAULT_MODEL})")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                        help=f"Early stopping patience (default: {DEFAULT_PATIENCE})")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    parser.add_argument("--export", type=str, choices=["onnx", "tflite", "coreml"], 
                        help="Export model format")
    parser.add_argument("--analyze", action="store_true", 
                        help="Show epoch analysis without training")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_only()
    elif args.validate:
        validate_model()
    elif args.export:
        export_model(args.export)
    else:
        train_model(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            base_model=args.model,
            patience=args.patience,
            resume=args.resume,
        )
