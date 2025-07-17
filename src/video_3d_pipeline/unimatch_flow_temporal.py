"""Temporal consistency enhancement using UniMatch optical flow."""

import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm

# Add UniMatch to path
project_root = Path(__file__).parent.parent.parent
unimatch_path = project_root / "unimatch"
sys.path.append(str(unimatch_path))

from unimatch.unimatch import UniMatch
from utils.utils import InputPadder
from dataloader.stereo import transforms

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class UniMatchFlowProcessor:
    """ Temporal consistency enhancement using UniMatch optical flow """
    
    def __init__(self,
                 flow_model_checkpoint: str = "./models/gmflow-scale2-regrefine6-things-776ed612.pth",
                 device: str = "cuda",
                 # Flow model parameters
                 num_scales: int = 2,
                 feature_channels: int = 128,
                 upsample_factor: int = 8,
                 num_head: int = 1,
                 ffn_dim_expansion: int = 4,
                 num_transformer_layers: int = 6,
                 reg_refine: bool = True,
                 num_reg_refine: int = 6,
                 attn_type: str = "self_swin2d_cross_1d",
                 attn_splits_list: List[int] = None,
                 corr_radius_list: List[int] = None,
                 prop_radius_list: List[int] = None,
                 padding_factor: int = 8):
        
        self.device = device
        self.flow_model_checkpoint = flow_model_checkpoint
        self.padding_factor = padding_factor
        
        # Flow model parameters
        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.num_head = num_head
        self.ffn_dim_expansion = ffn_dim_expansion
        self.num_transformer_layers = num_transformer_layers
        self.reg_refine = reg_refine
        self.num_reg_refine = num_reg_refine
        self.attn_type = attn_type
        self.attn_splits_list = attn_splits_list or [2]
        self.corr_radius_list = corr_radius_list or [-1]
        self.prop_radius_list = prop_radius_list or [-1]
        
        # Initialize models
        self.flow_model = None
        self.flow_model_loaded = False
        
        # Setup transforms
        self.val_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
        self.val_transform = transforms.Compose(self.val_transform_list)
        
        print(f"Initialized UniMatch Flow Processor")
        print(f"Device: {self.device}")
        print(f"Flow model: {self.flow_model_checkpoint}")
    
    def load_flow_model(self):
        """ Load UniMatch flow model """
        if self.flow_model_loaded:
            return
        
        print(f"Loading UniMatch flow model from {self.flow_model_checkpoint}...")
        
        # Initialize flow model
        self.flow_model = UniMatch(
            feature_channels=self.feature_channels,
            num_scales=self.num_scales,
            upsample_factor=self.upsample_factor,
            num_head=self.num_head,
            ffn_dim_expansion=self.ffn_dim_expansion,
            num_transformer_layers=self.num_transformer_layers,
            reg_refine=self.reg_refine,
            task='flow'  # Important: set task to 'flow'
        ).to(self.device)
        
        # Load checkpoint
        checkpoint_path = Path(self.flow_model_checkpoint)
        if not checkpoint_path.exists():
            raise ValueError(f"Flow checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            self.flow_model.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded flow checkpoint: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load flow checkpoint {checkpoint_path}: {e}")
        
        self.flow_model.eval()
        self.flow_model_loaded = True
        print("✓ UniMatch flow model loaded successfully")
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """ Preprocess single frame for flow computation """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Apply transforms
        sample = {'left': frame_rgb}  # Use 'left' key for single frame
        sample = self.val_transform(sample)
        
        # Move to device and add batch dimension
        frame_tensor = sample['left'].to(self.device).unsqueeze(0)  # [1, 3, H, W]
        
        return frame_tensor
    
    def compute_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """ Compute optical flow between two frames """
        if not self.flow_model_loaded:
            self.load_flow_model()
        
        with torch.no_grad():
            # Preprocess frames
            frame1_tensor = self.preprocess_frame(frame1)
            frame2_tensor = self.preprocess_frame(frame2)
            
            # Pad frames for model input
            padder = InputPadder(frame1_tensor.shape, padding_factor=self.padding_factor)
            frame1_padded, frame2_padded = padder.pad(frame1_tensor, frame2_tensor)
            
            # Compute flow
            flow_pred = self.flow_model(frame1_padded, frame2_padded,
                                      attn_type=self.attn_type,
                                      attn_splits_list=self.attn_splits_list,
                                      corr_radius_list=self.corr_radius_list,
                                      prop_radius_list=self.prop_radius_list,
                                      num_reg_refine=self.num_reg_refine,
                                      task='flow')['flow_preds'][-1]  # [1, 2, H, W]
            
            # Unpad result
            flow_pred = padder.unpad(flow_pred)[0]  # [2, H, W]
            
            # Convert to numpy and transpose to [H, W, 2]
            flow_np = flow_pred.cpu().numpy().transpose(1, 2, 0)
            
            return flow_np
    
    def warp_depth_with_flow(self, depth_map: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """ Warp depth map using optical flow """
        h, w = depth_map.shape
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply flow to get new coordinates
        new_x = x + flow[:, :, 0]
        new_y = y + flow[:, :, 1]
        
        # Clip coordinates to image bounds
        new_x = np.clip(new_x, 0, w - 1)
        new_y = np.clip(new_y, 0, h - 1)
        
        # Interpolate depth values at new coordinates
        warped_depth = cv2.remap(depth_map.astype(np.float32), 
                                new_x.astype(np.float32), 
                                new_y.astype(np.float32), 
                                cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=0)
        
        return warped_depth
    
    def detect_scene_change(self, frame1: np.ndarray, frame2: np.ndarray, 
                           threshold: float = 0.3) -> bool:
        """ Detect scene changes between frames using optical flow magnitude """
        flow = self.compute_optical_flow(frame1, frame2)
        
        # Compute flow magnitude
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        
        # Check if mean flow magnitude exceeds threshold
        mean_magnitude = np.mean(flow_magnitude)
        
        return mean_magnitude > threshold
    
    def enhance_temporal_consistency(self, 
                                   depth_maps: List[np.ndarray],
                                   frames: List[np.ndarray],
                                   alpha: float = 0.7,
                                   scene_change_threshold: float = 0.3) -> List[np.ndarray]:
        """ Enhance temporal consistency of depth maps using optical flow """
        if len(depth_maps) != len(frames):
            raise ValueError("Number of depth maps must match number of frames")
        
        if len(depth_maps) < 2:
            return depth_maps
        
        print("Enhancing temporal consistency with optical flow...")
        
        enhanced_depths = [depth_maps[0].copy()]  # First frame stays unchanged
        
        for i in tqdm(range(1, len(depth_maps)), desc="Flow consistency"):
            current_frame = frames[i]
            prev_frame = frames[i-1]
            current_depth = depth_maps[i]
            prev_depth = enhanced_depths[i-1]
            
            # Check for scene change
            if self.detect_scene_change(prev_frame, current_frame, scene_change_threshold):
                print(f"Scene change detected at frame {i}, skipping temporal update")
                enhanced_depths.append(current_depth.copy())
                continue
            
            # Compute optical flow from previous to current frame
            flow = self.compute_optical_flow(prev_frame, current_frame)
            
            # Warp previous depth map to current frame
            warped_prev_depth = self.warp_depth_with_flow(prev_depth, flow)
            
            # Blend warped previous depth with current depth
            # Use alpha blending with mask for valid warped regions
            valid_mask = (warped_prev_depth > 0) & (current_depth > 0)
            
            enhanced_depth = current_depth.copy()
            enhanced_depth[valid_mask] = (alpha * warped_prev_depth[valid_mask] + 
                                        (1 - alpha) * current_depth[valid_mask])
            
            enhanced_depths.append(enhanced_depth)
        
        print("✓ Temporal consistency enhancement complete")
        return enhanced_depths
    
    def compute_flow_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """ Compute optical flow for entire frame sequence """
        if len(frames) < 2:
            return []
        
        print("Computing optical flow sequence...")
        flows = []
        
        for i in tqdm(range(len(frames) - 1), desc="Computing flows"):
            flow = self.compute_optical_flow(frames[i], frames[i + 1])
            flows.append(flow)
        
        return flows
    
    def interpolate_depth_with_flow(self, 
                                  depth_maps: List[np.ndarray],
                                  frames: List[np.ndarray],
                                  target_fps_multiplier: int = 2) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """ Interpolate depth maps and frames using optical flow """
        if len(depth_maps) != len(frames):
            raise ValueError("Number of depth maps must match number of frames")
        
        if target_fps_multiplier <= 1:
            return depth_maps, frames
        
        print(f"Interpolating with {target_fps_multiplier}x FPS using optical flow...")
        
        interpolated_depths = []
        interpolated_frames = []
        
        for i in tqdm(range(len(frames) - 1), desc="Flow interpolation"):
            current_frame = frames[i]
            next_frame = frames[i + 1]
            current_depth = depth_maps[i]
            next_depth = depth_maps[i + 1]
            
            # Add current frame and depth
            interpolated_frames.append(current_frame)
            interpolated_depths.append(current_depth)
            
            # Compute forward and backward flows
            forward_flow = self.compute_optical_flow(current_frame, next_frame)
            backward_flow = self.compute_optical_flow(next_frame, current_frame)
            
            # Create intermediate frames and depths
            for t in range(1, target_fps_multiplier):
                alpha = t / target_fps_multiplier
                
                # Interpolate frame using flow
                # Warp current frame forward and next frame backward
                warped_current = self.warp_depth_with_flow(current_frame.astype(np.float32), 
                                                         forward_flow * alpha)
                warped_next = self.warp_depth_with_flow(next_frame.astype(np.float32), 
                                                      backward_flow * (1 - alpha))
                
                # Blend warped frames
                interpolated_frame = ((1 - alpha) * warped_current + alpha * warped_next).astype(np.uint8)
                interpolated_frames.append(interpolated_frame)
                
                # Interpolate depth using flow
                warped_current_depth = self.warp_depth_with_flow(current_depth, forward_flow * alpha)
                warped_next_depth = self.warp_depth_with_flow(next_depth, backward_flow * (1 - alpha))
                
                # Blend warped depths
                interpolated_depth = (1 - alpha) * warped_current_depth + alpha * warped_next_depth
                interpolated_depths.append(interpolated_depth)
        
        # Add last frame and depth
        interpolated_frames.append(frames[-1])
        interpolated_depths.append(depth_maps[-1])
        
        print(f"✓ Interpolated {len(frames)} frames to {len(interpolated_frames)} frames")
        return interpolated_depths, interpolated_frames


class TemporalDepthEnhancer:
    """ Combined stereo + flow pipeline for enhanced depth extraction """
    
    def __init__(self,
                 stereo_extractor,
                 flow_processor: UniMatchFlowProcessor = None,
                 temporal_alpha: float = 0.7,
                 scene_change_threshold: float = 0.3):
        
        self.stereo_extractor = stereo_extractor
        self.flow_processor = flow_processor or UniMatchFlowProcessor(device=stereo_extractor.device)
        self.temporal_alpha = temporal_alpha
        self.scene_change_threshold = scene_change_threshold
    
    def process_video_with_temporal_consistency(self,
                                              video_path: str,
                                              start_frame: int = 0,
                                              max_frames: int = None,
                                              force_reprocess: bool = False) -> Path:
        """ Process SBS video with temporal consistency enhancement """
        
        print("Starting enhanced depth extraction with temporal consistency...")
        
        # First, extract raw depth maps using stereo
        depth_cache_path = self.stereo_extractor.process_video_sbs(
            video_path=video_path,
            start_frame=start_frame,
            max_frames=max_frames,
            force_reprocess=force_reprocess
        )
        
        # Load extracted frames and depth maps
        frames = self.stereo_extractor.extract_frames_opencv(video_path, start_frame, max_frames)
        
        # Load depth maps
        depth_files = sorted(depth_cache_path.glob("depth_*.png"))
        depth_maps = []
        
        print("Loading depth maps for temporal processing...")
        for depth_file in tqdm(depth_files):
            depth_16bit = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            depth_map = depth_16bit.astype(np.float32) / 65535.0
            depth_maps.append(depth_map)
        
        # Extract left eye frames for flow computation
        left_frames = []
        for frame in frames:
            left, _ = self.stereo_extractor.split_sbs_frame(frame, unsqueeze=self.stereo_extractor.unsqueeze_sbs)
            left_frames.append(left)
        
        # Enhance temporal consistency
        enhanced_depths = self.flow_processor.enhance_temporal_consistency(
            depth_maps=depth_maps,
            frames=left_frames,
            alpha=self.temporal_alpha,
            scene_change_threshold=self.scene_change_threshold
        )
        
        # Save enhanced depth maps
        enhanced_cache_path = depth_cache_path.parent / f"{depth_cache_path.name}_temporal"
        enhanced_cache_path.mkdir(exist_ok=True)
        
        print("Saving temporally enhanced depth maps...")
        for i, enhanced_depth in enumerate(tqdm(enhanced_depths)):
            output_path = enhanced_cache_path / f"depth_{i:06d}.png"
            self.stereo_extractor.save_depth_map(enhanced_depth, output_path)
        
        print(f"✓ Enhanced depth extraction complete: {enhanced_cache_path}")
        return enhanced_cache_path