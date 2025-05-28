#!/usr/bin/env python3
import os
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
import motmetrics as mm
from collections import defaultdict

# ByteTrack implementation (simplified)
class BYTETracker(object):
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.tracked_tracks = []
        self.lost_tracks = []
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_id = 0
        self.max_time_lost = 30
        
    def update(self, dets, scores):
        self.frame_id += 1
        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []
        
        # Add high score detections as new tracks
        for i, (det, score) in enumerate(zip(dets, scores)):
            if score < self.track_thresh:
                continue
                
            # Check if it matches with any existing track
            matched = False
            for track in self.tracked_tracks:
                iou = self._iou(det, track['bbox'])
                if iou > self.match_thresh:
                    track['bbox'] = det
                    track['score'] = score
                    track['time_since_update'] = 0
                    matched = True
                    break
                    
            if not matched:
                # Create new track
                new_track = {
                    'bbox': det,
                    'score': score,
                    'track_id': len(self.tracked_tracks) + len(self.lost_tracks) + 1,
                    'time_since_update': 0
                }
                self.tracked_tracks.append(new_track)
                
        # Update existing tracks
        for track in self.tracked_tracks:
            if track['time_since_update'] > 0:
                track['time_since_update'] += 1
                
            if track['time_since_update'] > self.max_time_lost:
                self.lost_tracks.append(track)
                self.tracked_tracks.remove(track)
                
        # Return results
        results = []
        for track in self.tracked_tracks:
            if track['time_since_update'] == 0:
                x1, y1, x2, y2 = track['bbox']
                track_id = track['track_id']
                results.append(np.array([x1, y1, x2, y2, track_id]))
                
        return np.array(results)
    
    def _iou(self, bbox1, bbox2):
        """
        Compute IOU between two bounding boxes
        """
        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2
        
        # Calculate intersection area
        xx1 = max(x1, x1_)
        yy1 = max(y1, y1_)
        xx2 = min(x2, x2_)
        yy2 = min(y2, y2_)
        
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        inter = w * h
        
        # Calculate union area
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_ - x1_) * (y2_ - y1_)
        union = area1 + area2 - inter
        
        iou = inter / union if union > 0 else 0
        return iou

# Evaluation code
def read_gt_file(gt_file):
    """Read ground truth file"""
    gt_data = defaultdict(list)
    with open(gt_file, 'r') as f:
        for line in f:
            data = line.strip().split(',')
            if len(data) < 6:
                continue
            frame_id = int(data[0])
            track_id = int(data[1])
            x = float(data[2])
            y = float(data[3])
            w = float(data[4])
            h = float(data[5])
            gt_data[frame_id].append([x, y, x+w, y+h, track_id])
    return gt_data

def read_images(img_dir):
    """Read images from directory"""
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])
    return img_files

def detect_objects(image, detector):
    """Simple object detection (placeholder for actual detector)"""
    # For testing, we'll use the ground truth as detections
    # In a real scenario, you would use a detector like YOLO here
    h, w = image.shape[:2]
    # Return random detections for testing
    num_dets = np.random.randint(1, 10)
    dets = []
    scores = []
    for _ in range(num_dets):
        x1 = np.random.randint(0, w-100)
        y1 = np.random.randint(0, h-100)
        x2 = x1 + np.random.randint(50, 100)
        y2 = y1 + np.random.randint(50, 100)
        dets.append([x1, y1, x2, y2])
        scores.append(np.random.uniform(0.5, 1.0))
    return np.array(dets), np.array(scores)

def evaluate_sequence(seq_path, output_dir, gt_data=None):
    """Evaluate a sequence using ByteTrack"""
    img_dir = os.path.join(seq_path, 'img1')
    img_files = read_images(img_dir)
    
    # Initialize ByteTrack
    tracker = BYTETracker()
    
    # Output file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'results.txt')
    
    # Process each frame
    results = []
    for frame_idx, img_file in enumerate(tqdm(img_files, desc=f"Processing {os.path.basename(seq_path)}")):
        # Read image
        img = cv2.imread(img_file)
        
        # Detect objects
        if gt_data and frame_idx+1 in gt_data:
            # Use ground truth as detections for testing
            gt_boxes = np.array([box[:4] for box in gt_data[frame_idx+1]])
            gt_scores = np.ones(len(gt_boxes))
            dets, scores = gt_boxes, gt_scores
        else:
            dets, scores = detect_objects(img, None)
        
        # Update tracker
        tracks = tracker.update(dets, scores)
        
        # Save results
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            w, h = x2-x1, y2-y1
            line = f"{frame_idx+1},{int(track_id)},{x1},{y1},{w},{h},1,-1,-1,-1\n"
            with open(output_file, 'a') as f:
                f.write(line)
            
            # Draw on image
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"ID: {int(track_id)}", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save visualization
        vis_dir = os.path.join(output_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        cv2.imwrite(os.path.join(vis_dir, f"{frame_idx:06d}.jpg"), img)
    
    return output_file

def evaluate_mot_metrics(gt_file, result_file):
    """Evaluate tracking results using MOT metrics"""
    # Load ground truth
    gt = {}
    with open(gt_file, 'r') as f:
        for line in f:
            data = line.strip().split(',')
            if len(data) < 6:
                continue
            frame_id = int(data[0])
            track_id = int(data[1])
            x = float(data[2])
            y = float(data[3])
            w = float(data[4])
            h = float(data[5])
            if frame_id not in gt:
                gt[frame_id] = []
            gt[frame_id].append([x, y, w, h, track_id])
    
    # Load results
    results = {}
    with open(result_file, 'r') as f:
        for line in f:
            data = line.strip().split(',')
            if len(data) < 6:
                continue
            frame_id = int(data[0])
            track_id = int(data[1])
            x = float(data[2])
            y = float(data[3])
            w = float(data[4])
            h = float(data[5])
            if frame_id not in results:
                results[frame_id] = []
            results[frame_id].append([x, y, w, h, track_id])
    
    # Compute metrics
    acc = mm.MOTAccumulator(auto_id=True)
    
    for frame_id in sorted(gt.keys()):
        gt_dets = np.array([det[:4] for det in gt[frame_id]])
        gt_ids = np.array([det[4] for det in gt[frame_id]])
        
        if frame_id in results:
            res_dets = np.array([det[:4] for det in results[frame_id]])
            res_ids = np.array([det[4] for det in results[frame_id]])
        else:
            res_dets = np.empty((0, 4))
            res_ids = np.array([])
        
        # Compute distance matrix
        distances = mm.distances.iou_matrix(gt_dets, res_dets, max_iou=0.5)
        
        # Update accumulator
        acc.update(gt_ids, res_ids, distances)
    
    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'idf1', 'mostly_tracked', 
                                      'mostly_lost', 'num_false_positives', 'num_misses', 
                                      'num_switches', 'precision', 'recall'])
    
    return summary

def main():
    parser = argparse.ArgumentParser('ByteTrack evaluation script')
    parser.add_argument('--data_dir', type=str, default='/home/nail/Documents/MOTR/SportsMOT_example/dataset/train',
                        help='Path to SportsMOT dataset')
    parser.add_argument('--output_dir', type=str, default='/home/nail/Documents/MOTR/bytetrack_results',
                        help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get sequence directories
    seq_dirs = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    
    # Process each sequence
    for seq in seq_dirs:
        seq_path = os.path.join(args.data_dir, seq)
        seq_output_dir = os.path.join(args.output_dir, seq)
        
        # Read ground truth
        gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
        gt_data = read_gt_file(gt_file)
        
        # Evaluate sequence
        result_file = evaluate_sequence(seq_path, seq_output_dir, gt_data)
        
        # Compute metrics
        summary = evaluate_mot_metrics(gt_file, result_file)
        
        # Print results
        print(f"\nResults for sequence {seq}:")
        print(summary)
        
        # Save results
        summary.to_csv(os.path.join(seq_output_dir, 'metrics.csv'))
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
