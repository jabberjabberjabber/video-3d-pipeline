"""3D video conversion pipeline - extract depth from 3D and apply to 4K."""

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm

from .utils import get_video_info, create_work_directory


class StereoProcessor:
    """ Handles stereo correspondence and depth extraction """
    
    def __init__(self, block_size: int = 15, num_disparities: int = 64):
        self.block_size = block_size
        self.num_disparities = num_disparities
        
        # Initialize stereo matcher
        self.stereo = cv2.StereoBM_create(
            numDisparities=num_disparities,
            blockSize=block_size
        )
        
        # Parameters for better quality
        self.stereo.setPreFilterCap(63)
        self.stereo.setMinDisparity(0)
        self.stereo.setTextureThreshold(10)
        self.stereo.setUniquenessRatio(15)
        self.stereo.setSpeckleWindowSize(100)
        self.stereo.setSpeckleRange(32)

    def extract_depth_from_sbs(self, sbs_frame) -> np.ndarray:
        """ Extract depth map from side-by-side 3D frame """
        height, width = sbs_frame.shape[:2]
        mid_width = width // 2
        
        # Split into left and right views
        left = sbs_frame[:, :mid_width]
        right = sbs_frame[:, mid_width:]
        
        # Convert to grayscale if needed
        if len(left.shape) == 3:
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        else:
            left_gray, right_gray = left, right
            
        # Compute disparity
        disparity = self.stereo.compute(left_gray, right_gray)
        
        # Normalize to 0-255 range
        disparity_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        return disparity_norm

    def upscale_depth(self, depth_map, target_size: Tuple[int, int]) -> np.ndarray:
        """ Upscale depth map to target resolution """
        # Use edge-preserving upscaling
        upscaled = cv2.resize(depth_map, target_size, interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter to reduce upscaling artifacts
        filtered = cv2.bilateralFilter(upscaled, 9, 75, 75)
        
        return filtered


class DIBRRenderer:
    """ Depth-Image-Based Rendering for stereoscopic generation """
    
    def __init__(self, baseline: float = 6.5, focal_length: float = 1000):
        self.baseline = baseline  # Stereo baseline in cm
        self.focal_length = focal_length  # Focal length in pixels
        
    def render_stereo_pair(self, image, depth_map) -> Tuple[np.ndarray, np.ndarray]:
        """ Generate left/right views from 2D image and depth map """
        height, width = image.shape[:2]
        
        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert depth to disparity
        # Avoid division by zero
        depth_safe = np.maximum(depth_map.astype(np.float32), 1.0)
        disparity = (self.baseline * self.focal_length) / depth_safe
        
        # Generate left view (shift right)
        left_x = x_coords + disparity / 2
        left_image = self._warp_image(image, left_x, y_coords)
        
        # Generate right view (shift left)  
        right_x = x_coords - disparity / 2
        right_image = self._warp_image(image, right_x, y_coords)
        
        return left_image, right_image
    
    def _warp_image(self, image, x_coords, y_coords) -> np.ndarray:
        """ Warp image using coordinate maps """
        # Ensure coordinates are within bounds and float32
        x_coords = np.clip(x_coords, 0, image.shape[1] - 1).astype(np.float32)
        y_coords = np.clip(y_coords, 0, image.shape[0] - 1).astype(np.float32)
        
        # Use remap for warping
        warped = cv2.remap(image, x_coords, y_coords, cv2.INTER_LINEAR)
        
        return warped

    def create_sbs_output(self, left_image, right_image) -> np.ndarray:
        """ Combine left/right views into side-by-side format """
        return np.hstack([left_image, right_image])


class Video3DConverter:
    """ Main 3D conversion pipeline """
    
    def __init__(self, sbs_3d_path: str, hd_4k_path: str, work_dir: str = "temp_convert"):
        self.sbs_3d_path = sbs_3d_path
        self.hd_4k_path = hd_4k_path
        self.work_dir = create_work_directory(work_dir)
        
        # Get video information
        self.sbs_info = get_video_info(sbs_3d_path)
        self.hd_info = get_video_info(hd_4k_path)
        
        if not self.sbs_info or not self.hd_info:
            raise ValueError("Could not read video information")
            
        print(f"3D SBS: {self.sbs_info['width']}x{self.sbs_info['height']}")
        print(f"4K Source: {self.hd_info['width']}x{self.hd_info['height']}")
        
        # Initialize processors
        self.stereo_processor = StereoProcessor()
        self.dibr_renderer = DIBRRenderer()

    def process_video(self, output_path: str, max_frames: Optional[int] = None):
        """ Process entire video through 3D conversion pipeline """
        cap_sbs = cv2.VideoCapture(self.sbs_3d_path)
        cap_4k = cv2.VideoCapture(self.hd_4k_path)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.sbs_info['fps']
        output_size = (self.hd_info['width'] * 2, self.hd_info['height'])  # SBS output
        
        out = cv2.VideoWriter(output_path, fourcc, fps, output_size)
        
        frame_count = 0
        total_frames = min(
            int(cap_sbs.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap_4k.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
            
        print(f"Processing {total_frames} frames...")
        
        with tqdm(total=total_frames) as pbar:
            while frame_count < total_frames:
                ret_sbs, frame_sbs = cap_sbs.read()
                ret_4k, frame_4k = cap_4k.read()
                
                if not ret_sbs or not ret_4k:
                    break
                    
                # Extract depth from 3D frame
                depth_map = self.stereo_processor.extract_depth_from_sbs(frame_sbs)
                
                # Upscale depth to 4K resolution
                target_size = (self.hd_info['width'], self.hd_info['height'])
                depth_4k = self.stereo_processor.upscale_depth(depth_map, target_size)
                
                # Generate 4K stereo pair
                left_4k, right_4k = self.dibr_renderer.render_stereo_pair(frame_4k, depth_4k)
                
                # Create SBS output
                sbs_output = self.dibr_renderer.create_sbs_output(left_4k, right_4k)
                
                # Write frame
                out.write(sbs_output)
                
                frame_count += 1
                pbar.update(1)
        
        # Cleanup
        cap_sbs.release()
        cap_4k.release()
        out.release()
        
        print(f"✓ Conversion complete: {output_path}")


def main():
    """ Command line interface for 3D conversion """
    parser = argparse.ArgumentParser(description='Convert 1080p 3D to 4K 3D using depth extraction')
    parser.add_argument('sbs_3d', help='Path to 1080p side-by-side 3D video')
    parser.add_argument('hd_4k', help='Path to 4K 2D video') 
    parser.add_argument('--output', '-o', default='output_4k_3d.mp4', help='Output path')
    parser.add_argument('--max-frames', type=int, help='Limit processing to N frames (for testing)')
    parser.add_argument('--work-dir', default='temp_convert', help='Working directory')
    
    args = parser.parse_args()
    
    try:
        converter = Video3DConverter(args.sbs_3d, args.hd_4k, args.work_dir)
        converter.process_video(args.output, args.max_frames)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())