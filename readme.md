# Video 3D Pipeline

Convert stereoscopic 3D video releases to 4K 3D using depth extraction and upscaling.

## Overview

This project extracts depth information from existing stereoscopic 3D releases and applies it to 4K 2D versions of the same content to create high-quality 4K 3D video. **Supports all major 3D formats**: Side-by-Side, Top-Bottom, Anaglyph, and Interlaced.

### Why?

**Stereo depth extraction from existing 3D:**

✅ Real geometric depth from actual binocular disparity

✅ Sub-pixel accuracy from stereo correspondence

✅ Consistent depth across frames (no flickering)

✅ Fast processing - just computer vision algorithms

✅ **Memory efficient** - constant RAM usage regardless of video length

✅ **Storage efficient** - single compressed video output instead of thousands of files

**Monocular depth inference:**

❌ Guessed depth from single image cues

❌ Heavy neural network inference on every frame

❌ Temporal inconsistency between frames

❌ Slow GPU inference for large models

### Supported 3D Formats

| Format | Description | Example Use |
|--------|-------------|-------------|
| **Half SBS** | Side-by-side, compressed horizontally | Most common 3D format |
| **Full SBS** | Side-by-side, full resolution per eye | High-quality 3D releases |
| **Top-Bottom** | Vertically stacked, compressed vertically | 3D Blu-rays |
| **Anaglyph** | Red/cyan composite | Old-school 3D glasses |
| **Interlaced** | Line-by-line alternating | 3D TV broadcasts |

### Pipeline Steps

1. **Temporal Alignment** - Synchronize stereoscopic 3D and 4K 2D videos
2. **Depth Extraction** - Extract disparity maps from stereoscopic frames using UniMatch
3. **Depth Upscaling** - Upscale depth maps to 4K resolution
4. **3D Video Creation** - Combine 4K video with depth maps for native 4K 3D

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

### One-Command Pipeline (Recommended):

```bash
# Half SBS (most common format)
uv run python run_pipeline.py stereo_video.mp4 video_4k.mp4

# Full SBS (high quality)
uv run python run_pipeline.py stereo_video.mp4 video_4k.mp4 --stereo-format full_sbs

# Top-Bottom 3D
uv run python run_pipeline.py stereo_video.mp4 video_4k.mp4 --stereo-format top_bottom

# Red/Cyan Anaglyph
uv run python run_pipeline.py anaglyph_video.mp4 video_4k.mp4 --stereo-format anaglyph

# Interlaced 3D
uv run python run_pipeline.py interlaced_video.mp4 video_4k.mp4 --stereo-format interlaced

# Test with limited frames
uv run python run_pipeline.py stereo_video.mp4 video_4k.mp4 --max-frames 100

# Skip steps as needed
uv run python run_pipeline.py stereo_video.mp4 video_4k.mp4 --skip-alignment
```

### Step-by-Step Method:

```bash
# 1. Quick alignment (seconds)
uv run python -m video_3d_pipeline.align stereo_video.mp4 video_4k.mp4

# 2. Depth extraction with format selection
uv run python -m video_3d_pipeline.depth stereo_video.mp4 --stereo-format half_sbs

# 3. Upscale to 4K
uv run python -m video_3d_pipeline.upscale temp_depth/depth_video.mp4 video_4k.mp4

# 4. Use with VisionDepth3D
# - 4K video: video_4k.mp4
# - 4K depth: depth_4k_final.mp4
```

### Advanced Options:

```bash
# Memory optimization for large videos
uv run python run_pipeline.py stereo_video.mp4 video_4k.mp4 \
  --stereo-format half_sbs \
  --batch-size 4 \
  --inference-size 480 854

# Force reprocessing
uv run python run_pipeline.py stereo_video.mp4 video_4k.mp4 --force

# Custom work directory
uv run python run_pipeline.py stereo_video.mp4 video_4k.mp4 --work-dir custom_temp

# Disable aspect ratio restoration
uv run python run_pipeline.py stereo_video.mp4 video_4k.mp4 --no-unsqueeze
```

### Step 4: VisionDepth3D Integration

Now you can use VisionDepth3D with superior stereo-derived depth:

- **4K Video**: `video_4k.mp4`
- **4K Depth**: `temp_pipeline/depth_4k_final.mp4`

## Performance & Efficiency

### Memory Optimization
- **Streaming Processing**: Constant RAM usage regardless of video length
- **No Memory Crashes**: Process full-length movies without OOM errors
- **Batch Processing**: Configurable batch sizes for different GPU memory

### Storage Optimization  
- **Video Compression**: Single MP4 instead of thousands of PNG files
- **Space Savings**: 95%+ reduction in storage requirements
- **Fast I/O**: No filesystem bottlenecks from managing thousands of files

### Example Performance
```
173,140 frames (2-hour movie):
├── OLD: 50-100GB PNG files + 50GB RAM usage → OOM crash
└── NEW: 17MB MP4 file + 500MB RAM usage → Success! ✅
```

## Format-Specific Notes

### Half SBS vs Full SBS
- **Half SBS**: Most common, each eye compressed to 50% width
- **Full SBS**: Higher quality, double-width videos (e.g., 3840px → 1920px per eye)
- Use `--no-unsqueeze` to keep original compressed aspect ratios

### Top-Bottom
- Common on 3D Blu-ray releases
- Each eye compressed to 50% height
- Automatically restored to full aspect ratio unless `--no-unsqueeze`

### Anaglyph
- Works with classic red/cyan 3D glasses
- Left eye extracted from red channel, right eye from cyan channels
- Good for testing depth algorithms on old content

### Interlaced
- Used by 3D TV broadcasts and some displays
- Even lines = left eye, odd lines = right eye
- Includes line interpolation for missing data

## Troubleshooting

### Memory Issues
```bash
# Reduce batch size
uv run python run_pipeline.py video.mp4 4k.mp4 --batch-size 2

# Use smaller inference resolution
uv run python run_pipeline.py video.mp4 4k.mp4 --inference-size 360 640
```

### Storage Issues
```bash
# Use custom work directory
uv run python run_pipeline.py video.mp4 4k.mp4 --work-dir /path/to/large/drive

# Clean up previous runs
rm -rf temp_pipeline/
```

### Format Detection
```bash
# Test with limited frames first
uv run python run_pipeline.py video.mp4 4k.mp4 --max-frames 10 --stereo-format half_sbs

# Try different formats if depth looks wrong
uv run python run_pipeline.py video.mp4 4k.mp4 --stereo-format top_bottom
```

## Project Structure

```
video-3d-pipeline/
├── pyproject.toml             	# UV project configuration
├── README.md                  	# This file
├── src/video_3d_pipeline/
│   ├── __init__.py            	# Package initialization
│   ├── align.py          		# Video temporal alignment
│   ├── depth.py               	# IGEV depth extraction (backup)
│   ├── unimatch_depth.py      	# UniMatch depth extraction (main)
│   ├── upscale.py             	# Depth upscaling
│   └── utils.py               	# Shared utilities
├── models/				# Pretrained models
├── unimatch/				# UniMatch stereo matching
└── run_pipeline.py            	# Main pipeline tool
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
- **UniMatch** - State-of-the-art stereo matching
- **CUDA** - GPU acceleration

## Current Status

- ✅ **Audio-based Video Alignment**: Fast, accurate temporal synchronization
- ✅ **Multi-Format Depth Extraction**: GPU-accelerated processing for all 3D formats
- ✅ **Memory & Storage Optimization**: Streaming processing with video compression
- ✅ **Depth Upscaling**: Guided filtering for edge-preserving 4K depth maps
- 🔄 **DIBR**: Use DIBR to achieve stereo depth rendering

## Contributing

This is an experimental project for converting 3D video formats. Contributions welcome for:

- Additional stereo format support
- Improved stereo correspondence algorithms
- Better depth map upscaling techniques  
- Optimized DIBR rendering
- Quality assessment metrics
- Performance optimizations

## License

MIT License - see LICENSE file for details.