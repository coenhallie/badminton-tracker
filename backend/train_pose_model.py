#!/usr/bin/env python3
"""
Training script for the Badminton Pose/Stance Detection model.

This model detects and classifies badminton player stances:
- backhand-general
- defense
- lift
- offense
- serve
- smash

Dataset: badminton pose.v1i.yolo26 (1200 images from Roboflow)
"""

import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO


def setup_dataset():
    """
    Set up the dataset with correct absolute paths.
    Copies the dataset to the backend/datasets folder and updates paths.
    """
    # Source dataset location
    source_dataset = Path.home() / "Desktop" / "badminton pose.v1i.yolo26"
    
    # Target location in backend/datasets
    target_dataset = Path(__file__).parent / "datasets" / "badminton_pose.v1i.yolo26"
    
    if not source_dataset.exists():
        raise FileNotFoundError(f"Dataset not found at {source_dataset}")
    
    # Copy dataset if not already present
    if not target_dataset.exists():
        print(f"Copying dataset from {source_dataset} to {target_dataset}...")
        target_dataset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dataset, target_dataset, dirs_exist_ok=True)
        print("Dataset copied successfully!")
    else:
        print(f"Dataset already exists at {target_dataset}")
    
    # Create updated data.yaml with absolute paths
    data_yaml_path = target_dataset / "data.yaml"
    
    # Read original config
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths to be absolute
    config['train'] = str(target_dataset / "train" / "images")
    config['val'] = str(target_dataset / "valid" / "images")
    config['test'] = str(target_dataset / "test" / "images")
    
    # Write updated config
    with open(data_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated data.yaml with absolute paths")
    print(f"  - Train: {config['train']}")
    print(f"  - Val: {config['val']}")
    print(f"  - Test: {config['test']}")
    print(f"  - Classes ({config['nc']}): {config['names']}")
    
    return data_yaml_path


def train_pose_model(
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    model_size: str = "n",  # n=nano, s=small, m=medium, l=large, x=xlarge
    device: str = "mps",  # Use MPS for Apple Silicon, "cuda" for NVIDIA, "cpu" for CPU
    resume: bool = False,
):
    """
    Train the badminton pose detection model.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        model_size: YOLO model size (n, s, m, l, x)
        device: Device to train on (mps, cuda, cpu)
        resume: Whether to resume from last checkpoint
    """
    # Setup dataset and get data.yaml path
    data_yaml_path = setup_dataset()
    
    # Output directory for trained model
    output_dir = Path(__file__).parent / "models" / "badminton_pose"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    if resume and (output_dir / "weights" / "last.pt").exists():
        print("Resuming training from last checkpoint...")
        model = YOLO(str(output_dir / "weights" / "last.pt"))
    else:
        # Use YOLO26 detection model as base (consistent with application)
        model_name = f"yolo26{model_size}.pt"
        print(f"Loading base model: {model_name}")
        model = YOLO(model_name)
    
    # Training configuration
    print("\n" + "="*60)
    print("BADMINTON POSE MODEL TRAINING")
    print("="*60)
    print(f"Model: YOLO26{model_size.upper()}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    # Train the model
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=str(output_dir.parent),
        name="badminton_pose",
        exist_ok=True,
        patience=20,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,
        # Data augmentation
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=10.0,  # Rotation augmentation
        translate=0.1, # Translation augmentation
        scale=0.5,     # Scale augmentation
        shear=2.0,     # Shear augmentation
        flipud=0.0,    # No vertical flip (badminton is orientation sensitive)
        fliplr=0.5,    # Horizontal flip
        mosaic=1.0,    # Mosaic augmentation
        mixup=0.1,     # Mixup augmentation
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    
    # Validate the model
    print("\nRunning validation...")
    val_results = model.val()
    
    # Print results
    print("\nValidation Results:")
    print(f"  mAP50: {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")
    
    # Export to different formats
    print("\nExporting model...")
    
    # Export to ONNX for cross-platform deployment
    model.export(format="onnx", imgsz=img_size, simplify=True)
    print(f"  - ONNX exported")
    
    # Export to CoreML for iOS/macOS
    try:
        model.export(format="coreml", imgsz=img_size)
        print(f"  - CoreML exported")
    except Exception as e:
        print(f"  - CoreML export failed: {e}")
    
    # Copy best weights to a convenient location
    best_weights = output_dir / "weights" / "best.pt"
    if best_weights.exists():
        final_model_path = output_dir.parent / "badminton_pose_best.pt"
        shutil.copy(best_weights, final_model_path)
        print(f"\nBest model saved to: {final_model_path}")
    
    return results


def test_model():
    """Test the trained model on a sample image."""
    model_path = Path(__file__).parent / "models" / "badminton_pose" / "weights" / "best.pt"
    
    if not model_path.exists():
        print("No trained model found. Please train the model first.")
        return
    
    model = YOLO(str(model_path))
    
    # Test on validation images
    dataset_path = Path(__file__).parent / "datasets" / "badminton_pose.v1i.yolo26" / "valid" / "images"
    
    if dataset_path.exists():
        import random
        images = list(dataset_path.glob("*.jpg"))
        if images:
            test_image = random.choice(images)
            print(f"Testing on: {test_image}")
            
            results = model(str(test_image))
            
            for r in results:
                print(f"Detections: {len(r.boxes)}")
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    print(f"  - {class_name}: {conf:.2%}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Badminton Pose Detection Model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                        help="Model size (n=nano, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cuda, cpu)")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--test", action="store_true", help="Test model instead of training")
    
    args = parser.parse_args()
    
    if args.test:
        test_model()
    else:
        train_pose_model(
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            model_size=args.model,
            device=args.device,
            resume=args.resume,
        )
