# Video 3D Pipeline

Convert 1080p 3D video releases to 4K 3D using depth extraction and upscaling.

## Overview

This project extracts depth information from existing 1080p stereoscopic 3D releases and applies it to 4K 2D versions of the same content to create high-quality 4K 3D video.

### Pipeline Steps

1. **Temporal Alignment** - Synchronize 1080p 3D and 4K 2D videos
2. **Depth Extraction** - Extract disparity maps from stereoscopic frames  
3. **Depth Upscaling** - Upscale depth maps to 4K resolution
4. **DIBR Rendering** - Generate 4K stereoscopic output using Depth-Image-Based Rendering

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup project
git clone <repo-url>
cd video-3d-pipeline
uv sync

# Install in development mode
uv pip install -e .
```

## Usage

### Step 1: Align Videos

First, temporally align your 1080p 3D and 4K 2D videos using audio correlation:

```bash
uv run video-align input_1080p_3d.mkv input_4k_2d.mp4 --duration 60
```

This will:
- Extract and cache audio tracks from both videos
- Find optimal temporal alignment using audio cross-correlation
- Generate synchronized 60-second segments  
- Verify alignment quality using audio correlation
- Save results in `temp_alignment/` directory

**Audio-based sync advantages:**
- 10-100x faster than video-based methods
- Sub-second accuracy
- Works with different resolutions/quality
- Industry standard approach
- Much lower CPU usage

### Step 2: Convert to 4K 3D (Coming Soon)

```bash
uv run video-3d-convert temp_alignment/video1_aligned_60s.mp4 temp_alignment/video2_aligned_60s.mp4
```

## Project Structure

```
video-3d-pipeline/
├── pyproject.toml              # UV project configuration
├── README.md                   # This file
├── src/video_3d_pipeline/
│   ├── __init__.py            # Package initialization
│   ├── align.py               # Video temporal alignment
│   ├── convert.py             # 3D conversion pipeline (WIP)
│   └── utils.py               # Shared utilities
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
- 🚧 **Depth Extraction**: Stereo correspondence algorithms
- 🚧 **Depth Upscaling**: AI-based upscaling methods  
- 🚧 **DIBR Rendering**: 4K stereoscopic generation

## Contributing

This is an experimental project for converting 3D video formats. Contributions welcome for:

- Improved stereo correspondence algorithms
- Better depth map upscaling techniques  
- Optimized DIBR rendering
- Quality assessment metrics

## License

MIT License - see LICENSE file for details.