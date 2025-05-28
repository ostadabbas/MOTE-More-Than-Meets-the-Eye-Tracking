#!/usr/bin/env python3
import os
import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import motmetrics as mm
from collections import defaultdict

from ultralytics import YOLO

# Simple tracking function using IoU
def simple_track(detections, prev_tracks, iou_threshold=0.5, max_age=30):
    """Simple tracking based on IoU matching"""
    if not prev_tracks:
        # First frame, create new tracks for all detections
        new_tracks = []
        for i, det in enumerate(detections):
            new_tracks.append({
                'bbox': det[:4],
                'score': det[4],
                'track_id': i + 1,
                'age': 0
            })
        return new_tracks
    
    # Calculate IoU between current detections and previous tracks
    matched_indices = []
    unmatched_detections = list(range(len(detections)))
    unmatched_tracks = list(range(len(prev_tracks)))
    
    if len(detections) > 0 and len(prev_tracks) > 0:
        iou_matrix = np.zeros((len(detections), len(prev_tracks)))
        for d, det in enumerate(detections):
            for t, track in enumerate(prev_tracks):
                iou_matrix[d, t] = calculate_iou(det[:4], track['bbox'])
        
        # Use Hungarian algorithm for matching
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)
        
        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= iou_threshold:
                matched_indices.append((row, col))
                if row in unmatched_detections:
                    unmatched_detections.remove(row)
                if col in unmatched_tracks:
                    unmatched_tracks.remove(col)
    
    # Update matched tracks
    new_tracks = []
    for d, t in matched_indices:
        prev_tracks[t]['bbox'] = detections[d][:4]
        prev_tracks[t]['score'] = detections[d][4]
        prev_tracks[t]['age'] = 0
        new_tracks.append(prev_tracks[t])
    
    # Add new tracks for unmatched detections
    max_id = 0
    for track in prev_tracks:
        if track['track_id'] > max_id:
            max_id = track['track_id']
    
    for d in unmatched_detections:
        new_tracks.append({
            'bbox': detections[d][:4],
            'score': detections[d][4],
            'track_id': max_id + d + 1,
            'age': 0
        })
    
    # Update unmatched tracks
    for t in unmatched_tracks:
        if prev_tracks[t]['age'] < max_age:
            prev_tracks[t]['age'] += 1
            new_tracks.append(prev_tracks[t])
    
    return new_tracks

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / (box1_area + box2_area - intersection)

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

def evaluate_sequence(seq_path, output_dir, model):
    """Evaluate a sequence using simple tracking"""
    img_dir = os.path.join(seq_path, 'img1')
    img_files = read_images(img_dir)
    
    # Output file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'results.txt')
    
    # Clear previous results
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Process each frame
    tracks = []
    for frame_idx, img_file in enumerate(tqdm(img_files, desc=f"Processing {os.path.basename(seq_path)}")):
        # Read image
        img = cv2.imread(img_file)
        
        # Detect objects
        results = model(img, verbose=False)[0]
        
        # Convert detections to format expected by tracker
        dets = []
        for i, det in enumerate(results.boxes.data):
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            if cls == 0:  # Only track persons
                dets.append([x1, y1, x2, y2, conf])
        
        # Update tracker
        if len(dets) > 0:
            dets = np.array(dets)
            tracks = simple_track(dets, tracks)
            
            # Save results
            for t in tracks:
                if t['age'] == 0:  # Only save active tracks
                    x1, y1, x2, y2 = t['bbox']
                    w, h = x2-x1, y2-y1
                    
                    # Write to file
                    line = f"{frame_idx+1},{t['track_id']},{x1},{y1},{w},{h},1,-1,-1,-1\n"
                    with open(output_file, 'a') as f:
                        f.write(line)
                    
                    # Draw on image
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, f"ID: {t['track_id']}", (int(x1), int(y1)-10), 
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
        gt_dets = np.array([[det[0], det[1], det[0]+det[2], det[1]+det[3]] for det in gt[frame_id]])
        gt_ids = np.array([det[4] for det in gt[frame_id]])
        
        if frame_id in results:
            res_dets = np.array([[det[0], det[1], det[0]+det[2], det[1]+det[3]] for det in results[frame_id]])
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
    
    # Add HOTA metrics if available
    try:
        import hota_metrics
        summary = hota_metrics.add_hota_to_summary(summary, [acc], ['seq'])
    except ImportError:
        print("HOTA metrics module not available, skipping HOTA calculation")
    
    return summary

def main():
    parser = argparse.ArgumentParser('Simple tracking evaluation script')
    parser.add_argument('--data_dir', type=str, default='/home/nail/Documents/MOTR/SportsMOT_example/dataset/train',
                        help='Path to SportsMOT dataset')
    parser.add_argument('--output_dir', type=str, default='/home/nail/Documents/MOTR/bytetrack_results',
                        help='Output directory for results')
    parser.add_argument('--model', type=str, default='yolov8x.pt',
                        help='YOLO model to use. Options include: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt, yolov8x6.pt, yolo11n.pt, yolo11n-seg.pt, yolo11n-pose.pt')
    args = parser.parse_args()
    
    # Print available models
    print("\nAvailable Ultralytics models:")
    print("YOLOv8 models: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt, yolov8x6.pt")
    print("YOLO11 models: yolo11n.pt, yolo11n-seg.pt, yolo11n-pose.pt")
    print(f"Using model: {args.model}\n")
    
    # Load YOLO model
    try:
        model = YOLO(args.model)
        print(f"Successfully loaded {args.model}")
    except Exception as e:
        print(f"Error loading model {args.model}: {e}")
        print("Falling back to yolov8x.pt")
        model = YOLO("yolov8x.pt")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get sequence directories
    seq_dirs = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    
    # Process each sequence
    all_summaries = []
    for seq in seq_dirs:
        seq_path = os.path.join(args.data_dir, seq)
        seq_output_dir = os.path.join(args.output_dir, seq)
        
        # Read ground truth
        gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
        
        # Evaluate sequence
        result_file = evaluate_sequence(seq_path, seq_output_dir, model)
        
        # Compute metrics
        summary = evaluate_mot_metrics(gt_file, result_file)
        all_summaries.append(summary)
        
        # Print results
        print(f"\nResults for sequence {seq}:")
        print(summary)
        
        # Save results
        summary.to_csv(os.path.join(seq_output_dir, 'metrics.csv'))
    
    # Compute overall metrics
    if len(all_summaries) > 1:
        print("\nOverall results:")
        overall_summary = pd.concat(all_summaries)
        print(overall_summary)
        overall_summary.to_csv(os.path.join(args.output_dir, 'overall_metrics.csv'))
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
