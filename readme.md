# Video 3D Pipeline

Convert 1080p 3D video releases to 4K 3D using depth extraction and upscaling.

## Overview

This project extracts depth information from existing 1080p stereoscopic 3D releases and applies it to 4K 2D versions of the same content to create high-quality 4K 3D video.

### Why?

**Stereo depth extraction from existing 3D:**

✅ Real geometric depth from actual binocular disparity

✅ Sub-pixel accuracy from stereo correspondence

✅ Consistent depth across frames (no flickering)

✅ Fast processing - just computer vision algorithms


**Monocular depth inference:**

❌ Guessed depth from single image cues

❌ Heavy neural network inference on every frame

❌ Temporal inconsistency between frames

❌ Slow GPU inference for large models

### Pipeline Steps

1. **Temporal Alignment** - Synchronize 1080p 3D and 4K 2D videos
2. **Depth Extraction** - Extract disparity maps from stereoscopic frames  
3. **Depth Upscaling** - Upscale depth maps to 4K resolution
4. **3D Video Creation with VisionDepth3D** - Add created depth map along with 4K video to create a native 4K 3D video

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup project
git clone <repo-url>
cd video-3d-pipeline
uv sync
```

## Usage

### Fast Method (Recommended):
```bash
# 1. Quick alignment (seconds)
uv run python -m video_3d_pipeline.align sbs_1080p.mp4 video_4k.mp4

# 2. Depth extraction with aspect fix
uv run python -m video_3d_pipeline.depth sbs_1080p.mp4

# 3. Upscale to 4K
uv run python -m video_3d_pipeline.upscale temp_depth/depth_maps/ video_4k.mp4

# 4. Use with VisionDepth3D
# - 4K video: video_4k.mp4
# - 4K depth: depth_4k_depth_*.mp4
```

### One-Command Pipeline:
```bash
# Complete pipeline
uv run python run_pipeline.py sbs_1080p.mp4 video_4k.mp4

# Test with limited frames
uv run python run_pipeline.py sbs_1080p.mp4 video_4k.mp4 --max-frames 100

# Skip steps as needed
uv run python run_pipeline.py sbs_1080p.mp4 video_4k.mp4 --skip-alignment
```

### Step 4: VisionDepth3D Integration

Now you can use VisionDepth3D with superior stereo-derived depth.

## Project Structure

```
video-3d-pipeline/
├── pyproject.toml             # UV project configuration
├── README.md                  # This file
├── src/video_3d_pipeline/
│   ├── __init__.py            # Package initialization
│   ├── align.py          # Video temporal alignment
│   ├── depth.py               # Depth extraction using hybrid stereo
│   ├── upscale.py             # Depth upscaling with guided filtering
│   └── utils.py               # Shared utilities
└── run_pipeline.py            # Main tool
```

## Dependencies

- **OpenCV** - Computer vision and video processing
- **FFmpeg** - Video encoding/decoding via ffmpeg-python
- **librosa** - Audio analysis and processing
- **SciPy** - Audio cross-correlation algorithms
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization and plotting
- **scikit-image** - Image processing algorithms
- **PyTorch** - Deep learning framework for GPU acceleration
- **Transformers** - CREStereo model loading and inference
- **CUDA** - GPU compute capability (optional but recommended)

## Current Status

- ✅ **Audio-based Video Alignment**: Fast, accurate temporal synchronization using cross-correlation
- ✅ **Depth Extraction**: GPU-accelerated hybrid stereo processing with neural guidance
- ✅ **Depth Upscaling**: Guided filtering for edge-preserving 4K depth map upscaling
- 🔄 **VisionDepth3D Integration**: Use existing DIBR pipeline with superior stereo depth

## Contributing

This is an experimental project for converting 3D video formats. Contributions welcome for:

- Improved stereo correspondence algorithms
- Better depth map upscaling techniques  
- Optimized DIBR rendering
- Quality assessment metrics

## License

MIT License - see LICENSE file for details.