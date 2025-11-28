#!/usr/bin/env python3
"""
Simple script to export YOLOv8 model to ONNX format
"""
import sys
import os

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "yolo8n_384_dg_micro-mAP-52_12_08_2025.pt"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print("Exporting to ONNX format...")
    model.export(
        format="onnx",
        imgsz=384,
        simplify=True,
        opset=12,
        dynamic=False,
        half=False
    )
    
    # Find the exported ONNX file
    onnx_file = model_path.replace(".pt", ".onnx")
    if os.path.exists(onnx_file):
        print(f"✓ ONNX model exported to {onnx_file}")
        if output_dir != ".":
            import shutil
            output_file = os.path.join(output_dir, os.path.basename(onnx_file))
            shutil.copy(onnx_file, output_file)
            print(f"✓ Copied to {output_file}")
    else:
        print(f"✗ ONNX export failed. File not found: {onnx_file}")
        sys.exit(1)

