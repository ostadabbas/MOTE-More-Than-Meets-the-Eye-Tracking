#!/usr/bin/env python
# ------------------------------------------------------------------------
# MOTR Demo with Disocclusion Matrix Visualization
# ------------------------------------------------------------------------

import os
import sys
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
from PIL import Image

# Add RAFT core directory to path
sys.path.append('/home/nail/Documents/MOTR/RAFT/core')

from models.motr_splatted import build as build_model_splatted
from util.tool import load_model
from main import get_args_parser
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

# Set up colors for visualization
COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0)]

def save_flow_visualization(flow_tensor, save_path):
    # Import flow visualization utilities from RAFT
    import sys
    sys.path.append('/home/nail/Documents/MOTR/RAFT/core')
    from RAFT.core.utils import flow_viz
    
    # Convert tensor to numpy array
    if isinstance(flow_tensor, torch.Tensor):
        # RAFT outputs flow in format [B, 2, H, W]
        # Convert to [H, W, 2] for visualization
        flow_np = flow_tensor[0].permute(1, 2, 0).cpu().numpy()
    else:
        # If already numpy array, ensure correct format
        flow_np = flow_tensor
        if flow_np.shape[0] == 2 and len(flow_np.shape) == 3:
            # Convert from [2, H, W] to [H, W, 2]
            flow_np = flow_np.transpose(1, 2, 0)
    
    # Convert flow to RGB using RAFT's visualization
    flow_rgb = flow_viz.flow_to_image(flow_np)
    
    # Save visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(flow_rgb)
    plt.title('Optical Flow (RAFT)')
    plt.savefig(save_path)
    plt.close()
    
    return flow_rgb

def save_disocclusion_visualization(disocclusion_map, save_path):
    """Save disocclusion matrix visualization"""
    # Normalize for visualization
    disocclusion_norm = (disocclusion_map - np.min(disocclusion_map)) / (np.max(disocclusion_map) - np.min(disocclusion_map) + 1e-6)
    
    # Create a colormap using red tones
    plt.figure(figsize=(10, 8))
    plt.imshow(disocclusion_norm, cmap='Reds')
    plt.colorbar()
    plt.title('Disocclusion Matrix')
    plt.savefig(save_path)
    plt.close()
    
    return disocclusion_norm

def create_combined_visualization(frame, flow_rgb, disocclusion_map, save_path):
    """Create a combined visualization with optical flow and disocclusion matrix side by side"""
    # Normalize disocclusion map for visualization
    disocclusion_norm = (disocclusion_map - np.min(disocclusion_map)) / (np.max(disocclusion_map) - np.min(disocclusion_map) + 1e-6)
    
    # Convert disocclusion map to RGB using a red colormap
    # Create a custom red colormap that goes from black to red
    cmap = plt.cm.Reds
    disocclusion_rgb = cmap(disocclusion_norm)
    disocclusion_rgb = (disocclusion_rgb[:, :, :3] * 255).astype(np.uint8)
    
    # Get dimensions
    if isinstance(flow_rgb, np.ndarray):
        target_height = flow_rgb.shape[0]
    else:
        # If flow_rgb is not a numpy array, convert it
        flow_rgb = np.array(flow_rgb)
        target_height = flow_rgb.shape[0]
    
    # Resize disocclusion_rgb if needed
    if disocclusion_rgb.shape[0] != target_height:
        disocclusion_rgb = cv2.resize(disocclusion_rgb, (int(disocclusion_rgb.shape[1] * target_height / disocclusion_rgb.shape[0]), target_height))
    
    # Create combined image (flow and disocclusion only, no original frame)
    combined_img = np.hstack((flow_rgb, disocclusion_rgb))
    
    # Save the combined visualization
    cv2.imwrite(save_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
    
    return combined_img

class ImageSequenceProcessor:
    def __init__(self, args):
        # Build model and load weights
        self.model, _, _ = build_model_splatted(args)
        
        # Custom model loading to handle class mismatch
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        # Remove class_embed keys from the checkpoint to avoid shape mismatch
        model_state_dict = checkpoint['model']
        for k in list(model_state_dict.keys()):
            if 'class_embed' in k:
                del model_state_dict[k]
        
        # Load the modified state dict
        self.model.load_state_dict(model_state_dict, strict=False)
        self.model = self.model.cpu()  # Use CPU instead of GPU
        self.model.eval()
        
        # Store the data path
        self.data_path = args.mot17_data_path
        
        # Create output directories
        self.output_dir = args.output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.flow_dir = os.path.join(self.output_dir, 'flow')
        Path(self.flow_dir).mkdir(parents=True, exist_ok=True)
        
        self.disocclusion_dir = os.path.join(self.output_dir, 'disocclusion')
        Path(self.disocclusion_dir).mkdir(parents=True, exist_ok=True)
        
        self.frames_dir = os.path.join(self.output_dir, 'frames')
        Path(self.frames_dir).mkdir(parents=True, exist_ok=True)
        
        self.combined_dir = os.path.join(self.output_dir, 'combined')
        Path(self.combined_dir).mkdir(parents=True, exist_ok=True)

    def process_sequence(self):
        # Get all image files in the directory
        image_files = sorted([f for f in os.listdir(self.data_path) if f.endswith('.jpg')])
        
        if not image_files:
            print(f"No image files found in {self.data_path}")
            return
        
        # Read first frame
        prev_frame_path = os.path.join(self.data_path, image_files[0])
        prev_frame = cv2.imread(prev_frame_path)
        
        # Save first frame
        cv2.imwrite(os.path.join(self.frames_dir, f'frame_0.jpg'), prev_frame)
        
        # Convert to tensor
        prev_frame_tensor = self.preprocess_frame(prev_frame)
        
        # Process sequence
        total_frames = len(image_files)
        print(f"Processing {total_frames} frames from MOT17 sequence")
        
        with torch.no_grad():
            for frame_idx in range(1, min(total_frames, 100)):  # Limit to 100 frames for testing
                # Read current frame
                curr_frame_path = os.path.join(self.data_path, image_files[frame_idx])
                curr_frame = cv2.imread(curr_frame_path)
                
                # Save current frame
                cv2.imwrite(os.path.join(self.frames_dir, f'frame_{frame_idx}.jpg'), curr_frame)
                
                # Convert to tensor
                curr_frame_tensor = self.preprocess_frame(curr_frame)
                
                # Calculate optical flow
                try:
                    optical_flow = self.model.optical_flow_module(prev_frame_tensor, curr_frame_tensor)
                    
                    # Save optical flow visualization
                    flow_np = optical_flow[0].permute(1, 2, 0).cpu().numpy()
                    flow_rgb = save_flow_visualization(flow_np, 
                                                     os.path.join(self.flow_dir, f'flow_{frame_idx}.jpg'))
                    
                    # Extract features and calculate disocclusion matrix
                    features = []
                    with torch.no_grad():
                        # Create a NestedTensor for the backbone
                        from util.misc import NestedTensor
                        
                        # Resize the frame to the expected input size
                        dummy_samples = torch.nn.functional.interpolate(
                            curr_frame_tensor.unsqueeze(0), 
                            size=(800, 1536), 
                            mode='bilinear', 
                            align_corners=False
                        )
                        
                        # Create a mask (all False)
                        mask = torch.zeros(dummy_samples.shape[0], dummy_samples.shape[2], dummy_samples.shape[3], 
                                          dtype=torch.bool)
                        
                        # Create a NestedTensor
                        nested_tensor = NestedTensor(dummy_samples, mask)
                        
                        # Get features from backbone
                        features_out, _ = self.model.backbone(nested_tensor)
                        
                        # Extract feature tensors and print dimensions for debugging
                        for i, feat in enumerate(features_out):
                            src, _ = feat.decompose()
                            print(f"Feature level {i} shape: {src.shape}")
                            features.append(src)
                    
                    # Use softmax splatting to get disocclusion matrix
                    splatted_features = self.model.softmax_splatting(features, optical_flow)
                    
                    # Visualize disocclusion matrix (using first channel as example)
                    disocclusion_map = splatted_features[0, 0].cpu().numpy()
                    disocclusion_norm = save_disocclusion_visualization(
                        disocclusion_map, 
                        os.path.join(self.disocclusion_dir, f'disocclusion_{frame_idx}.jpg')
                    )
                    
                    # Create combined visualization (frame, flow, disocclusion)
                    create_combined_visualization(
                        curr_frame,  # Original frame
                        flow_rgb,     # Flow visualization
                        disocclusion_map,  # Disocclusion map
                        os.path.join(self.combined_dir, f'combined_{frame_idx}.jpg')
                    )
                    
                    print(f"Processed frame {frame_idx}/{total_frames}")
                    
                    # Create a video from combined visualizations every 50 frames
                    if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
                        self.create_visualization_video(frame_idx)
                    
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                
                # Update for next iteration
                prev_frame_tensor = curr_frame_tensor
        
        print(f"Processing complete. Results saved to {self.output_dir}")
    
    def preprocess_frame(self, frame):
        """Convert OpenCV frame to PyTorch tensor"""
        # Convert to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Convert to float and normalize
        frame_float = frame_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor [C, H, W]
        frame_tensor = torch.from_numpy(frame_float).permute(2, 0, 1)
        
        return frame_tensor
        
    def create_visualization_video(self, current_frame_idx):
        """Create a video from the combined visualizations up to the current frame"""
        # Check if we have enough frames to create a video
        if current_frame_idx < 10:
            return
            
        # Get all combined visualization images
        combined_files = sorted([f for f in os.listdir(self.combined_dir) if f.endswith('.jpg') and f.startswith('combined_')])
        
        if not combined_files:
            print("No combined visualization files found.")
            return
            
        # Get the first image to determine dimensions
        first_img = cv2.imread(os.path.join(self.combined_dir, combined_files[0]))
        height, width, layers = first_img.shape
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = os.path.join(self.output_dir, f'visualization_up_to_frame_{current_frame_idx}.avi')
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
        
        # Add each image to the video
        for file in combined_files:
            if int(file.split('_')[1].split('.')[0]) <= current_frame_idx:
                img = cv2.imread(os.path.join(self.combined_dir, file))
                out.write(img)
        
        # Release the VideoWriter
        out.release()
        
        print(f"Created visualization video: {video_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MOTR with Disocclusion Matrix Demo', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # Set required parameters
    args.mot17_data_path = '/home/nail/Documents/MOTR/data/MOT17/train/MOT17-02-DPM/img1'
    args.resume = '/home/nail/Documents/MOTR/model_weights/model_motr_final.pth'
    args.with_box_refine = True
    args.meta_arch = 'motr_splatted'  # Use the splatted version
    args.sampler_lengths = [2, 3, 4, 5]
    args.num_classes = 91  # Set to 91 to match the pre-trained model
    args.dataset_file = 'coco'  # Set to coco to match the pre-trained model
    args.output_dir = '/home/nail/Documents/MOTR/output_splatted_mot17'
    
    processor = ImageSequenceProcessor(args)
    processor.process_sequence()
