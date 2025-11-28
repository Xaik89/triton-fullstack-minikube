#!/usr/bin/env bash
# Generate inference requests to produce real metrics for Grafana (GPU util, latency, etc.)
# Defaults to 10 requests; use --forever or --count N to control volume.

set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
  -n, --count N     Number of requests to send (default: 10). Use 0 with --forever.
      --forever     Run indefinitely (Ctrl+C to stop).
  -s, --sleep S     Seconds to sleep between requests (default: 2).
  -u, --url URL     Target endpoint (default: http://localhost:8000/cv/detect).
  -i, --image PATH  Image to send (default: create /tmp/test_image.png if missing).
  -h, --help        Show this help.
EOF
}

REQUEST_LIMIT=10
SLEEP_SECONDS=2
TARGET_URL="http://localhost:8000/cv/detect"
IMAGE_PATH="/tmp/test_image.png"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--count) REQUEST_LIMIT="${2:-}"; shift 2 ;;
    --forever) REQUEST_LIMIT=0; shift ;;
    -s|--sleep) SLEEP_SECONDS="${2:-}"; shift 2 ;;
    -u|--url) TARGET_URL="${2:-}"; shift 2 ;;
    -i|--image) IMAGE_PATH="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ "$REQUEST_LIMIT" =~ ^[0-9]+$ ]]; then
  :
else
  echo "Invalid --count value: $REQUEST_LIMIT" >&2
  exit 1
fi

echo "Generating inference requests to create real metrics..."
if (( REQUEST_LIMIT == 0 )); then
  echo "Mode: forever (Ctrl+C to stop)"
else
  echo "Mode: $REQUEST_LIMIT request(s)"
fi
echo ""

# Connectivity hint for local Minikube users
if ! curl -s -o /dev/null --connect-timeout 2 "$TARGET_URL"; then
  echo "Cannot reach $TARGET_URL" >&2
  echo "If using Minikube, port-forward first:" >&2
  echo "  kubectl port-forward -n triton-inference svc/backend 8000:8000" >&2
  echo "Or pass the service URL:" >&2
  echo "  $(command -v minikube >/dev/null 2>&1 && minikube service backend -n triton-inference --url 2>/dev/null | head -n1 || echo \"--url http://<backend-host>:8000/cv/detect\")" >&2
  exit 1
fi

# Create a simple test image (384x384 pixels, required by YOLOv8n model)
if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Creating test image (384x384) at $IMAGE_PATH..."
  if ! python3 -c "from PIL import Image" 2>/dev/null; then
    echo "Error: Pillow library not found. Install with: pip install Pillow" >&2
    echo "Or supply your own image via --image /path/to/file" >&2
    exit 1
  fi
  python3 << EOF
from PIL import Image
img = Image.new('RGB', (384, 384), color='white')
img.save('$IMAGE_PATH')
print("Test image created at $IMAGE_PATH")
EOF
elif [[ ! -r "$IMAGE_PATH" ]]; then
  echo "Image not readable: $IMAGE_PATH" >&2
  exit 1
fi

COUNTER=0
trap 'echo; echo "Stopped after $COUNTER request(s)."' INT

while :; do
  if (( REQUEST_LIMIT > 0 && COUNTER >= REQUEST_LIMIT )); then
    echo "Completed $COUNTER request(s)."
    break
  fi

  COUNTER=$((COUNTER + 1))
  echo "Request #$COUNTER"
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$TARGET_URL" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@${IMAGE_PATH}") || {
      echo "curl failed on request #$COUNTER" >&2
      exit 1
    }
  echo "Status: $STATUS"
  sleep "$SLEEP_SECONDS"
done
