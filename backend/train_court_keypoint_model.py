#!/usr/bin/env python3
"""
Train a custom YOLO-Pose model for badminton court KEYPOINT detection.
Uses the "BadmintonCourtDetectionOffical" dataset from Roboflow.

This model detects 22 court keypoints for precise court localization:
- Court corners (4 outer corners)
- Service line intersections
- Net line positions
- Center line points
- All key court geometry points

EPOCH ANALYSIS:
===============
Dataset: 918 training images, 202 validation images, 74 test images
Total: 1,194 images
Keypoints: 22 per court with visibility flags
Task: Pose/Keypoint detection (more complex than object detection)

Recommendation: 150-200 epochs with early stopping (patience=50)

For keypoint detection:
- Keypoint localization requires more training than bounding boxes
- Smaller dataset (~1,200 images) needs more epochs for convergence
- 22 keypoints with geometric relationships require precise learning
- Early stopping with patience=50 will find optimal stopping point
- Default 150 epochs with early stopping is recommended

COMPARISON WITH BOUNDING BOX COURT DETECTION:
- Bounding box model: ~2,582 images, 8 classes, 100 epochs
- Keypoint model: ~918 images, 22 keypoints, 150 epochs (recommended)
- Keypoint detection provides PRECISE court localization for mini-court mapping

Usage:
    cd backend
    source venv/bin/activate
    python train_court_keypoint_model.py

    # With custom epochs
    python train_court_keypoint_model.py --epochs 200

    # Analyze epoch requirements only
    python train_court_keypoint_model.py --analyze

The trained model will be saved in models/court_keypoint/weights/best.pt
"""

from ultralytics import YOLO
import os
import argparse
import json
from datetime import datetime

# Configuration
DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets", "BadmintonCourtDetectionOffical.v1i.yolov8")
DATA_YAML = os.path.join(DATASET_PATH, "data.yaml")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")

# Default training parameters optimized for court keypoint detection
# Recommended: 150-200 epochs for keypoint detection with ~1,200 training images
DEFAULT_EPOCHS = 150  # Higher than object detection due to keypoint complexity
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_MODEL = "yolo26n-pose.pt"  # Pose model for keypoint detection
DEFAULT_PATIENCE = 50  # Higher patience - keypoint models need more time to converge


def analyze_epoch_requirements():
    """
    Analyze and print recommendations for optimal epoch count.
    
    Based on:
    - Dataset size (~1,194 total images)
    - Number of keypoints (22 per court)
    - Nature of detection (geometric keypoints requiring precise localization)
    """
    print("\n" + "=" * 70)
    print("COURT KEYPOINT DETECTION - EPOCH ANALYSIS & RECOMMENDATIONS")
    print("=" * 70)
    
    analysis = {
        "dataset_size": {
            "train": 918,
            "valid": 202,
            "test": 74,
            "total": 1194
        },
        "keypoints": 22,
        "keypoint_format": "[x, y, visibility] per keypoint",
        "detection_type": "Pose/Keypoint detection (court geometry)",
        "recommended_epochs": "150-200",
        "optimal_starting_epochs": 150,
        "early_stopping_patience": 50,
        "comparison": {
            "bounding_box_model": {
                "dataset_size": 2582,
                "classes": 8,
                "recommended_epochs": "80-120",
                "task": "Object detection"
            },
            "keypoint_model": {
                "dataset_size": 918,
                "keypoints": 22,
                "recommended_epochs": "150-200",
                "task": "Pose/Keypoint detection"
            }
        },
        "reasoning": [
            "Keypoint detection requires more epochs than bounding box detection",
            "Smaller dataset (~1,200 images) needs additional epochs for convergence",
            "22 keypoints with complex geometric relationships require precise learning",
            "Pose models typically need 50-100% more epochs than object detection",
            "Early stopping with patience=50 prevents overfitting automatically",
            "If early stopping triggers before epoch 100, consider reducing patience",
            "If training completes all epochs without early stopping, increase to 200"
        ],
        "keypoint_purposes": [
            "Outer court corners (4 points) - Court boundary",
            "Inner court corners (4 points) - Service areas",
            "Service line intersections (6 points) - Service boxes",
            "Net line positions (4 points) - Net area",
            "Center line points (4 points) - Court center",
            "Additional keypoints for precise perspective transformation"
        ]
    }
    
    print(f"\nDataset Statistics:")
    print(f"  Training images: {analysis['dataset_size']['train']}")
    print(f"  Validation images: {analysis['dataset_size']['valid']}")
    print(f"  Test images: {analysis['dataset_size']['test']}")
    print(f"  Total: {analysis['dataset_size']['total']} images")
    print(f"\nKeypoint Configuration:")
    print(f"  Keypoints per court: {analysis['keypoints']}")
    print(f"  Format: {analysis['keypoint_format']}")
    print(f"\nTask Complexity:")
    print(f"  Detection type: {analysis['detection_type']}")
    print(f"  Recommended Epochs: {analysis['recommended_epochs']}")
    print(f"  Starting with: {analysis['optimal_starting_epochs']} epochs")
    print(f"  Early Stopping Patience: {analysis['early_stopping_patience']}")
    
    print(f"\nComparison with Bounding Box Court Detection:")
    bb = analysis['comparison']['bounding_box_model']
    kp = analysis['comparison']['keypoint_model']
    print(f"  Bounding Box Model: {bb['dataset_size']} images, {bb['classes']} classes, {bb['recommended_epochs']} epochs")
    print(f"  Keypoint Model: {kp['dataset_size']} images, {kp['keypoints']} keypoints, {kp['recommended_epochs']} epochs")
    
    print("\nReasoning:")
    for i, reason in enumerate(analysis['reasoning'], 1):
        print(f"  {i}. {reason}")
    
    print("\nKeypoint Purposes (22 points):")
    for purpose in analysis['keypoint_purposes']:
        print(f"  • {purpose}")
    
    print("=" * 70 + "\n")
    
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
    Train the YOLO-Pose model on the court keypoint detection dataset.
    
    Args:
        epochs: Number of training epochs (recommended: 150-200)
        imgsz: Image size for training
        batch: Batch size
        base_model: Base YOLO-Pose model to fine-tune
        patience: Early stopping patience (recommended: 50)
        resume: Whether to resume from last checkpoint
    """
    # Show epoch analysis
    analyze_epoch_requirements()
    
    print("=" * 70)
    print("BADMINTON COURT KEYPOINT DETECTION MODEL TRAINING")
    print("=" * 70)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Data config: {DATA_YAML}")
    print(f"Base model: {base_model}")
    print(f"Epochs: {epochs}")
    print(f"Early Stopping Patience: {patience}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print("=" * 70)
    
    # Verify dataset exists
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"Dataset config not found: {DATA_YAML}\n"
            f"Please ensure the dataset is copied to: {DATASET_PATH}"
        )
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load base pose model
    print(f"\nLoading base pose model: {base_model}")
    model = YOLO(base_model)
    
    # Train the model with settings optimized for court keypoint detection
    print("\nStarting training...")
    print(f"Training will stop early if no improvement for {patience} epochs")
    print("Monitor the pose metrics - keypoint detection typically aims for:")
    print("  - mAP50: >0.75 for good keypoint localization")
    print("  - OKS (Object Keypoint Similarity): Higher is better\n")
    
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=OUTPUT_DIR,
        name="court_keypoint",
        exist_ok=True,
        resume=resume,
        # Performance optimizations
        workers=4,
        patience=patience,  # Early stopping - critical for finding optimal epochs
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        # Data augmentation optimized for court keypoint detection
        hsv_h=0.01,    # Minimal hue augmentation (court colors are consistent)
        hsv_s=0.3,     # Light saturation augmentation
        hsv_v=0.4,     # Value augmentation (important for different lighting)
        degrees=5.0,   # Light rotation (courts have fixed orientation)
        translate=0.1, # Light translation
        scale=0.3,     # Scale augmentation for different camera angles
        shear=2.0,     # Light shear for perspective variations
        perspective=0.0005,  # Light perspective augmentation
        flipud=0.0,    # No vertical flip (courts have fixed orientation)
        fliplr=0.5,    # Horizontal flip (courts are symmetric)
        mosaic=0.5,    # Reduced mosaic for keypoint detection
        mixup=0.0,     # No mixup for geometric keypoint detection
        copy_paste=0.0,  # No copy-paste for pose detection
        # Loss weights for keypoint detection
        box=7.5,       # Box loss weight
        cls=0.5,       # Classification loss weight  
        dfl=1.5,       # Distribution focal loss
        pose=12.0,     # Pose/keypoint loss weight (important for keypoint accuracy)
        kobj=1.0,      # Keypoint objectness loss
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    # Find the best weights
    best_weights = os.path.join(OUTPUT_DIR, "court_keypoint", "weights", "best.pt")
    
    # Training summary
    if hasattr(results, 'results_dict'):
        print("\nTraining Summary:")
        print(f"  Epochs completed: {results.epoch}")
        if results.box:
            print(f"  Best mAP50 (boxes): {results.box.map50:.4f}")
            print(f"  Best mAP50-95 (boxes): {results.box.map:.4f}")
        if hasattr(results, 'pose') and results.pose:
            print(f"  Best mAP50 (pose): {results.pose.map50:.4f}")
            print(f"  Best mAP50-95 (pose): {results.pose.map:.4f}")
    
    # Report early stopping status
    results_csv = os.path.join(OUTPUT_DIR, "court_keypoint", "results.csv")
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
                print(f"  → Consider increasing epochs if metrics are still improving")
    
    if os.path.exists(best_weights):
        print(f"\nBest model saved to: {best_weights}")
        print("\nTo use this model for court keypoint detection, update your .env file:")
        print(f"  YOLO_COURT_KEYPOINT_MODEL={best_weights}")
        print("\nOr to compare with bounding box model, you can run inference with both:")
        print("  - Bounding box model: models/court/weights/best.pt")
        print("  - Keypoint model: models/court_keypoint/weights/best.pt")
    
    # Save training metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "model_type": "pose/keypoint",
        "epochs_requested": epochs,
        "patience": patience,
        "base_model": base_model,
        "imgsz": imgsz,
        "batch": batch,
        "dataset": DATASET_PATH,
        "keypoints": 22,
        "classes": 1,
        "description": "Badminton court keypoint detection for precise court localization"
    }
    metadata_path = os.path.join(OUTPUT_DIR, "court_keypoint", "training_metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return results


def validate_model():
    """Validate the trained model on the test set."""
    model_path = os.path.join(OUTPUT_DIR, "court_keypoint", "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        print("Run training first: python train_court_keypoint_model.py")
        return
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    print("Running validation on test set...")
    results = model.val(data=DATA_YAML, split="test")
    
    print("\nValidation Results:")
    print(f"  Box mAP50: {results.box.map50:.4f}")
    print(f"  Box mAP50-95: {results.box.map:.4f}")
    
    if hasattr(results, 'pose') and results.pose:
        print(f"  Pose mAP50: {results.pose.map50:.4f}")
        print(f"  Pose mAP50-95: {results.pose.map:.4f}")
    
    return results


def compare_models():
    """Compare keypoint model with bounding box model."""
    keypoint_model_path = os.path.join(OUTPUT_DIR, "court_keypoint", "weights", "best.pt")
    bbox_model_path = os.path.join(OUTPUT_DIR, "court", "weights", "best.pt")
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: Keypoint vs Bounding Box Court Detection")
    print("=" * 70)
    
    models_found = []
    
    if os.path.exists(keypoint_model_path):
        print(f"\n✓ Keypoint model found: {keypoint_model_path}")
        models_found.append(("keypoint", keypoint_model_path))
    else:
        print(f"\n✗ Keypoint model not found: {keypoint_model_path}")
        print("  Run: python train_court_keypoint_model.py")
    
    if os.path.exists(bbox_model_path):
        print(f"✓ Bounding box model found: {bbox_model_path}")
        models_found.append(("bbox", bbox_model_path))
    else:
        print(f"✗ Bounding box model not found: {bbox_model_path}")
        print("  Run: python train_court_model.py")
    
    if len(models_found) == 2:
        print("\nBoth models are available for comparison!")
        print("\nKey differences:")
        print("  Keypoint Model:")
        print("    - Outputs 22 precise keypoints on the court")
        print("    - Better for perspective transformation (mini-court mapping)")
        print("    - More accurate for court boundary detection")
        print("  Bounding Box Model:")
        print("    - Outputs 8 region bounding boxes")
        print("    - Faster inference")
        print("    - Better for simple court presence detection")
        print("\nTo compare performance, run validation on both models with your test video.")
    
    return models_found


def export_model(format: str = "onnx"):
    """Export the trained model to different formats."""
    model_path = os.path.join(OUTPUT_DIR, "court_keypoint", "weights", "best.pt")
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
        description="Train YOLO-Pose model for badminton court keypoint detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EPOCH RECOMMENDATIONS FOR KEYPOINT DETECTION:
  - Default: 150 epochs with early stopping (patience=50)
  - Minimum recommended: 120 epochs
  - Maximum recommended: 200 epochs (if no early stopping triggers)
  - Early stopping will automatically find optimal stopping point

KEYPOINT VS BOUNDING BOX COMPARISON:
  - Keypoint model: 22 precise court keypoints, better for mapping
  - Bounding box model: 8 region boxes, faster but less precise

EXAMPLES:
  python train_court_keypoint_model.py                    # Default 150 epochs
  python train_court_keypoint_model.py --epochs 200      # More epochs
  python train_court_keypoint_model.py --analyze         # Show epoch analysis
  python train_court_keypoint_model.py --validate        # Validate trained model
  python train_court_keypoint_model.py --compare         # Compare with bbox model
"""
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, 
                        help=f"Number of epochs (default: {DEFAULT_EPOCHS}, recommended: 150-200)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, 
                        help=f"Batch size (default: {DEFAULT_BATCH})")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, 
                        help=f"Image size (default: {DEFAULT_IMGSZ})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                        help=f"Base pose model (default: {DEFAULT_MODEL})")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                        help=f"Early stopping patience (default: {DEFAULT_PATIENCE})")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    parser.add_argument("--compare", action="store_true", help="Compare with bounding box model")
    parser.add_argument("--export", type=str, choices=["onnx", "tflite", "coreml"], 
                        help="Export model format")
    parser.add_argument("--analyze", action="store_true", 
                        help="Show epoch analysis without training")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_only()
    elif args.validate:
        validate_model()
    elif args.compare:
        compare_models()
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
