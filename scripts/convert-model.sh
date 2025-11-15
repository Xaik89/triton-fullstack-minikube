#!/bin/bash

set -e

echo "=== Converting YOLOv8n Model to TensorRT ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

MODEL_PATH="$HOME/data/DG/models/yolo8n_384_dg_micro-mAP-52_12_08_2025.pt"
OUTPUT_DIR="$PROJECT_DIR/triton-cv/models/yolo8n/1"
OUTPUT_FILE="$OUTPUT_DIR/model.plan"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found at: $MODEL_PATH"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: ./setup-venv.sh first"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_FILE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Convert model (export to ONNX + build TensorRT engine)
python3 - << PY
import os

print("CUDA_VISIBLE_DEVICES before:", os.environ.get("CUDA_VISIBLE_DEVICES"))
# Ensure GPU 0 is visible to TensorRT
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

from ultralytics import YOLO
import tensorrt as trt

model_path = "${MODEL_PATH}"
engine_path = "${OUTPUT_FILE}"
onnx_path = model_path.replace(".pt", ".onnx")
input_size = (384, 384)

print(f"Loading YOLOv8n model from {model_path}...")
model = YOLO(model_path)

print("Exporting to ONNX format...")
model.export(
    format="onnx",
    imgsz=input_size,
    simplify=True,
    opset=12,
    dynamic=False,
    half=False,
)

if not os.path.exists(onnx_path):
    raise SystemExit(f"ONNX export failed, file not found: {onnx_path}")

print(f"ONNX model exported to {onnx_path}")
print("Converting ONNX to TensorRT engine...")

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        print("ERROR: Failed to parse ONNX file")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit(1)

config = builder.create_builder_config()
try:
    from tensorrt import MemoryPoolType
    config.set_memory_pool_limit(MemoryPoolType.WORKSPACE, 1 << 30)
except (AttributeError, ImportError):
    config.max_workspace_size = 1 << 30  # type: ignore[attr-defined]

print("Building TensorRT engine (this may take a while)...")
serialized_engine = None
try:
    serialized_engine = builder.build_serialized_network(network, config)
except AttributeError:
    engine = builder.build_engine(network, config)  # type: ignore[attr-defined]
    if engine is not None:
        serialized_engine = engine.serialize()

if serialized_engine is None:
    raise SystemExit("ERROR: Failed to build TensorRT engine")

os.makedirs(os.path.dirname(engine_path), exist_ok=True)
with open(engine_path, "wb") as f:
    f.write(serialized_engine)

print(f"TensorRT engine saved to {engine_path}")
PY

if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "✓ Model converted successfully!"
    echo "  Location: $OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE"
else
    echo ""
    echo "❌ Conversion failed!"
    exit 1
fi
