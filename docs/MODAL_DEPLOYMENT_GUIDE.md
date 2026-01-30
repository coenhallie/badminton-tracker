# Modal.com Deployment Guide for Badminton Tracker

## Overview

This guide explains how to deploy the YOLO inference pipeline to Modal.com for GPU-accelerated processing. Moving inference from a local MacBook to Modal provides:

| Metric | Local MacBook (M1/M2) | Modal (T4 GPU) | Modal (A10G GPU) |
|--------|----------------------|----------------|------------------|
| Pose Inference | ~150-200ms | ~15-25ms | ~8-15ms |
| Object Detection | ~100-150ms | ~10-15ms | ~5-10ms |
| Court Detection | ~100-150ms | ~10-15ms | ~5-10ms |
| **Total per Frame** | **~400-500ms** | **~35-55ms** | **~20-35ms** |
| Cost | Free (your hardware) | ~$0.000164/sec | ~$0.000356/sec |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Local Machine                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    FastAPI Backend                           │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │    │
│  │  │   Upload    │  │  Analytics  │  │   Post-Processing   │  │    │
│  │  │   Handler   │  │   Module    │  │   (Smoothing, etc)  │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │    │
│  │                         │                                    │    │
│  │                    ModalClient                               │    │
│  │                         │                                    │    │
│  └─────────────────────────│────────────────────────────────────┘    │
└────────────────────────────│─────────────────────────────────────────┘
                             │ HTTPS (Base64 frames)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Modal.com Cloud                               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                 BadmintonInference Class                     │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │    │
│  │  │ Pose Model  │  │  Detection  │  │    Court Model      │  │    │
│  │  │ (YOLOv8n)   │  │   Model     │  │   (Custom YOLO)     │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │    │
│  │                         GPU: T4/A10G/A100                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │               Modal Volume (Persistent Storage)              │    │
│  │  /root/models/court/best.pt                                  │    │
│  │  /root/models/badminton/best.pt                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Python 3.10+**: Required for Modal CLI
3. **Modal CLI**: Installed via pip

## Step-by-Step Deployment

### 1. Install Modal CLI

```bash
# Install Modal
pip install modal

# Authenticate (opens browser)
modal setup
```

### 2. Deploy to Modal

```bash
# From the backend directory
cd backend

# Deploy the inference service
modal deploy modal_inference.py

# Note the endpoint URL in the output, e.g.:
# ✓ Created web endpoint at https://your-workspace--badminton-tracker-inference.modal.run
```

### 3. Upload Custom Models (Optional)

If you have custom trained models for court detection or badminton detection:

```bash
# Upload court detection model
modal run modal_inference.py::upload_model \
  --model-type court \
  --local-path ./models/court/weights/best.pt

# Upload badminton detection model
modal run modal_inference.py::upload_model \
  --model-type badminton \
  --local-path ./models/badminton/weights/best.pt
```

### 4. Configure Local Backend

Add to your `.env` file:

```env
# Enable Modal inference
USE_MODAL_INFERENCE=true

# Modal endpoint URL (from step 2)
MODAL_ENDPOINT_URL=https://your-workspace--badminton-tracker-inference.modal.run

# Optional: Increase timeout for large frames
MODAL_TIMEOUT=30.0
```

### 5. Test the Deployment

```bash
# Test from command line
modal run modal_inference.py

# Or test the health endpoint
curl https://your-workspace--badminton-tracker-inference.modal.run/health
```

### 6. Start Local Backend

```bash
# The backend will automatically use Modal for inference
python -m uvicorn main:app --reload
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MODAL_INFERENCE` | `false` | Enable/disable Modal inference |
| `MODAL_ENDPOINT_URL` | `` | Modal web endpoint URL |
| `MODAL_API_KEY` | `` | Optional API key for auth |
| `MODAL_TIMEOUT` | `30.0` | Request timeout (seconds) |
| `MODAL_MAX_RETRIES` | `3` | Max retry attempts |

### GPU Selection

In `modal_inference.py`, you can change the GPU type:

```python
@app.cls(
    gpu="T4",      # Cost-effective, good performance
    # gpu="A10G",  # Faster, ~2x cost of T4
    # gpu="A100",  # Fastest, ~4x cost of T4
    ...
)
```

GPU Pricing (approximate, pay-per-second):
- **T4**: $0.59/hour (~$0.000164/sec)
- **A10G**: $1.28/hour (~$0.000356/sec)
- **A100-40GB**: $3.74/hour (~$0.001039/sec)

### Scaling Configuration

```python
@app.cls(
    gpu="T4",
    timeout=300,                    # Max 5 minutes per request
    container_idle_timeout=120,     # Keep warm for 2 minutes
    allow_concurrent_inputs=10,     # Handle 10 concurrent requests
)
```

## Monitoring & Debugging

### View Logs

```bash
# Stream logs from Modal dashboard
modal app logs badminton-tracker-inference
```

### Check Volume Contents

```bash
# List files in the volume
modal volume ls badminton-tracker-models
```

### Force Container Restart

If models need to be reloaded:

```bash
modal app stop badminton-tracker-inference
modal deploy modal_inference.py
```

## Hybrid Mode

The `HybridInferenceManager` in `modal_client.py` automatically handles:

1. **Modal First**: Tries Modal inference when available
2. **Automatic Fallback**: Falls back to local CPU if Modal fails
3. **Graceful Degradation**: Continues processing even with errors

```python
from modal_client import HybridInferenceManager

manager = HybridInferenceManager()
await manager.initialize()  # Returns "modal" or "local"

# Process frame - uses best available backend
result = await manager.process_frame(frame)
print(f"Processed via: {result['inference_source']}")
```

## Cost Estimation

For a typical badminton video analysis:

| Video Length | Frames (30fps) | Modal T4 Time | Estimated Cost |
|--------------|---------------|---------------|----------------|
| 1 minute | ~1,800 | ~50-90 sec | $0.008-0.015 |
| 5 minutes | ~9,000 | ~250-450 sec | $0.04-0.07 |
| 30 minutes | ~54,000 | ~25-45 min | $0.25-0.45 |

*Note: Actual costs depend on frame processing settings, batch sizes, and container cold starts.*

## Troubleshooting

### "Connection refused" Error

Modal container may be cold. Wait ~10-30 seconds for warmup or:

```bash
# Keep container warm with periodic health checks
while true; do
  curl -s https://your-endpoint/health
  sleep 60
done
```

### "Out of Memory" Error

Reduce batch size or upgrade GPU:

```python
# In modal_inference.py
@app.cls(gpu="A10G")  # More VRAM than T4
```

### Timeout Errors

Increase timeout in `.env`:

```env
MODAL_TIMEOUT=60.0
```

### Model Not Loading

Check if models are uploaded to Volume:

```bash
modal volume ls badminton-tracker-models
# Should show:
# /court/best.pt
# /badminton/best.pt
```

## Security Considerations

1. **API Key**: For production, add authentication:

   ```python
   # In modal_inference.py
   @modal.fastapi_endpoint(method="POST", docs=True, auth="token")
   def infer_frame(request: FrameRequest) -> FrameResult:
       ...
   ```

2. **Rate Limiting**: Modal handles this automatically

3. **Data Privacy**: Frames are transmitted over HTTPS, not stored

## Next Steps

1. **Deploy to production**: `modal deploy --env production modal_inference.py`
2. **Set up monitoring**: Use Modal's dashboard for metrics
3. **Optimize**: Experiment with different GPU types
4. **Scale**: Modal auto-scales based on demand

## References

- [Modal Documentation](https://modal.com/docs)
- [Modal YOLO Example](https://modal.com/docs/examples/finetune_yolo)
- [Ultralytics YOLO26](https://docs.ultralytics.com/)
