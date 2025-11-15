#!/usr/bin/env python3
"""
Convert YOLOv8n model to TensorRT engine for Triton Inference Server
"""
import os
import sys
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
    import torch
    import tensorrt as trt
    import numpy as np
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install: ultralytics, torch, tensorrt")
    sys.exit(1)


def export_to_onnx(model_path: str, output_path: str, input_size: tuple = (640, 640)):
    """
    Export YOLOv8n model to ONNX format
    """
    print(f"Loading YOLOv8n model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Exporting to ONNX format...")
    model.export(
        format="onnx",
        imgsz=input_size,
        simplify=True,
        opset=12,
        dynamic=False,
        half=False  # Use FP32 for compatibility
    )
    
    # Find the exported ONNX file
    onnx_file = model_path.replace(".pt", ".onnx")
    if not os.path.exists(onnx_file):
        # Try alternative naming
        onnx_file = output_path
    
    if os.path.exists(onnx_file):
        print(f"ONNX model exported to {onnx_file}")
        return onnx_file
    else:
        raise FileNotFoundError(f"ONNX export failed. Expected file: {onnx_file}")


def onnx_to_tensorrt(onnx_path: str, engine_path: str, input_shape: tuple = (1, 3, 640, 640)):
    """
    Convert ONNX model to TensorRT engine
    """
    print(f"Converting ONNX to TensorRT engine...")
    
    # Create TensorRT logger
    logger = trt.Logger(trt.Logger.WARNING)
    
    # Create builder and network
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    # Configure builder
    config = builder.create_builder_config()

    # Set workspace / memory pool size (TensorRT 10+ API with fallback)
    try:
        # New API (TensorRT 10+)
        from tensorrt import MemoryPoolType
        config.set_memory_pool_limit(MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    except (AttributeError, ImportError):
        # Older API (< TensorRT 10)
        config.max_workspace_size = 1 << 30  # type: ignore[attr-defined]
    
    # Build serialized engine (TensorRT 10+) with fallback to legacy API
    print("Building TensorRT engine (this may take a while)...")
    serialized_engine = None
    engine = None

    try:
        # Preferred in TensorRT 10+
        serialized_engine = builder.build_serialized_network(network, config)
    except AttributeError:
        # Legacy path for older TensorRT versions
        engine = builder.build_engine(network, config)  # type: ignore[attr-defined]
        if engine is not None:
            serialized_engine = engine.serialize()
    
    if serialized_engine is None:
        print("ERROR: Failed to build TensorRT engine")
        return False
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"TensorRT engine saved to {engine_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert YOLOv8n to TensorRT")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                       help="Path to YOLOv8n PyTorch model (will download if not exists)")
    parser.add_argument("--output", type=str, default="model.plan",
                       help="Output path for TensorRT engine")
    parser.add_argument("--input-size", type=int, nargs=2, default=[384, 384],
                       help="Input size [height width]")
    
    args = parser.parse_args()
    
    # Download model if it doesn't exist
    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}, downloading YOLOv8n...")
        model = YOLO("yolov8n.pt")
        args.model = "yolov8n.pt"
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Export to ONNX
    onnx_path = args.output.replace(".plan", ".onnx")
    onnx_path = export_to_onnx(args.model, onnx_path, tuple(args.input_size))
    
    # Step 2: Convert ONNX to TensorRT
    input_shape = (1, 3, args.input_size[0], args.input_size[1])
    success = onnx_to_tensorrt(onnx_path, args.output, input_shape)
    
    if success:
        print(f"\n✓ Conversion complete!")
        print(f"  TensorRT engine: {args.output}")
        print(f"  Input shape: {input_shape}")
    else:
        print("\n✗ Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
