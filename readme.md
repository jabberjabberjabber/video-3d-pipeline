# Video 3D Pipeline

Convert 1080p 3D video releases to 4K 3D using depth extraction and upscaling.

## Overview

This project extracts depth information from existing 1080p stereoscopic 3D releases and applies it to 4K 2D versions of the same content to create high-quality 4K 3D video.

### Pipeline Steps

1. **Temporal Alignment** - Synchronize 1080p 3D and 4K 2D videos
2. **Depth Extraction** - Extract disparity maps from stereoscopic frames  
3. **Depth Upscaling** - Upscale depth maps to 4K resolution

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

### Step 1: Align Videos

First, temporally align your 1080p 3D and 4K 2D videos using audio correlation:

```bash
uv run video-align input_1080p_3d.mkv input_4k_2d.mp4 --duration 60
```

### Step 2: Extract Depth Maps

Extract depth information from the aligned 1080p 3D video using CREStereo:

```bash
uv run video-depth-extract temp_alignment/video1_aligned_60s.mp4 --batch-size 8 --model /models/
```

**GPU requirements:**
- CUDA-capable GPU with 6GB+ VRAM
- Automatic batch size optimization based on available memory
- Dynamic VRAM management to prevent overflow

### Step 3: Upscale Depth Maps to 4K

Upscale the 1080p depth maps to 4K resolution using the 4K 2D video as guidance:

```bash
uv run video-depth-upscale temp_depth/depth_maps_folder/ temp_alignment/video2_aligned_60s.mp4
```

### Step 4: VisionDepth3D Integration

Now you can use VisionDepth3D with superior stereo-derived depth.

## Project Structure

```
video-3d-pipeline/
├── pyproject.toml              # UV project configuration
├── README.md                   # This file
├── src/video_3d_pipeline/
│   ├── __init__.py            # Package initialization
│   ├── align.py               # Video temporal alignment
│   ├── depth.py               # Depth extraction using hybrid stereo
│   ├── upscale.py             # Depth upscaling with guided filtering
│   ├── convert.py             # 3D conversion pipeline (deprecated)
│   └── utils.py               # Shared utilities
├── examples/
│   └── full_pipeline.py       # Complete workflow example
└── tests/                     # Test suite (TBD)
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

## Quality Expectations

- **Audio correlation >0.8**: Excellent temporal sync
- **Audio correlation 0.6-0.8**: Good sync, suitable for processing  
- **Audio correlation <0.6**: May need manual adjustment or different source videos
- **Video compatibility check**: Verifies duration and frame rate compatibility

## Development

```bash
# Install development dependencies
uv sync --extra dev

# Run linting
uv run ruff check src/
uv run black src/

# Run tests (when implemented)
uv run pytest
```

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