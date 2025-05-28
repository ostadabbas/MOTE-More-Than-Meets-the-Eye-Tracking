#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
    SportsMOT Evaluation Script for MOTR
"""
from __future__ import print_function

import os
import numpy as np
import random
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
from models import build_model
from util.tool import load_model
from main import get_args_parser
from torch.nn.functional import interpolate
from typing import List, Dict
from util.evaluation import Evaluator
import motmetrics as mm
import shutil
import hota_metrics

from models.structures import Instances

np.random.seed(2020)

COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]


def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = 2
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        if score is not None:
            cv2.putText(img, score, (c1[0], c1[1] + 30), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def draw_bboxes(ori_img, bbox, identities=None, offset=(0, 0), cvt_color=False):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
        else:
            score = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label = '{:d}'.format(id)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        img = plot_one_box([x1, y1, x2, y2], img, color, label, score=score)
    return img


def draw_points(img: np.ndarray, points: np.ndarray, color=(255, 255, 255)):
    assert len(points.shape) == 2 and points.shape[1] == 2, 'invalid points shape: {}'.format(points.shape)
    for i, (x, y) in enumerate(points):
        if i >= 300:
            color = (0, 255, 0)
        cv2.circle(img, (int(x), int(y)), 2, color=color, thickness=2)
    return img


def tensor_to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


class Track(object):
    track_cnt = 0

    def __init__(self, box):
        self.box = box
        self.time_since_update = 0
        self.id = Track.track_cnt
        Track.track_cnt += 1
        self.miss = 0

    def miss_one_frame(self):
        self.miss += 1

    def clear_miss(self):
        self.miss = 0

    def update(self, box):
        self.box = box
        self.clear_miss()


class MOTR(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        pass

    def update(self, dt_instances: Instances):
        ret = []
        for i in range(len(dt_instances)):
            label = dt_instances.labels[i].item() if dt_instances.has('labels') else 0
            if label == 0:  # person class
                bbox = dt_instances.boxes[i]
                score = dt_instances.scores[i].item() if dt_instances.has('scores') else 1.0
                track_id = dt_instances.obj_idxes[i].item() if dt_instances.has('obj_idxes') else -1
                ret.append(np.array([bbox[0], bbox[1], bbox[2], bbox[3], score, track_id]))
        if len(ret) > 0:
            return np.stack(ret)
        return np.array([])


def load_label(label_path: str, img_size: tuple):
    def scale_box(bbox, w, h, img_size):
        x, y, width, height = bbox
        x *= img_size[1]
        width *= img_size[1]
        y *= img_size[0]
        height *= img_size[0]
        return [x, y, width, height]

    labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            values = line.strip().split(' ')
            if len(values) < 5:
                continue
            cls, x, y, w, h = values[0:5]
            labels.append(scale_box([float(x), float(y), float(w), float(h)], float(w), float(h), img_size))
    return labels


class SportsMOTEvaluator(Evaluator):
    """
    Extended Evaluator class for SportsMOT dataset
    """
    def __init__(self, data_root, seq_name, data_type='mot'):
        super().__init__(data_root, seq_name, data_type)
        
    def load_annotations(self):
        """Modified to handle SportsMOT format"""
        assert self.data_type == 'mot'
        
        gt_filename = os.path.join(self.data_root, self.seq_name, 'gt', 'gt.txt')
        self.gt_frame_dict = self.read_sportsmot_results(gt_filename, is_gt=True)
        self.gt_ignore_frame_dict = self.read_sportsmot_results(gt_filename, is_ignore=True)
    
    def read_sportsmot_results(self, filename, is_gt=False, is_ignore=False):
        """Read SportsMOT format results"""
        results_dict = dict()
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                for line in f.readlines():
                    linelist = line.split(',')
                    if len(linelist) < 6:
                        continue
                    fid = int(linelist[0])
                    if fid < 1:
                        continue
                    results_dict.setdefault(fid, list())

                    if is_gt:
                        # In SportsMOT, all entries in gt.txt are valid
                        score = 1
                    elif is_ignore:
                        # No ignore regions in SportsMOT
                        continue
                    else:
                        score = 1.0  # No score in SportsMOT results
                    
                    # Format: <frame_id>,<track_id>,<x>,<y>,<w>,<h>,<conf>,<class>,<visibility>
                    # For SportsMOT, we use x,y,w,h format
                    tlwh = tuple(map(float, linelist[2:6]))
                    target_id = int(linelist[1])
                    results_dict[fid].append((tlwh, target_id, score))
        
        return results_dict


class Detector(object):
    def __init__(self, args, model=None, seq_num=None):
        self.args = args
        self.detr = model
        self.seq_num = seq_num
        
        # Get image list from dataset
        img_list = os.listdir(os.path.join(self.args.mot_path, 'dataset/train', self.seq_num, 'img1'))
        img_list = [os.path.join(self.args.mot_path, 'dataset/train', self.seq_num, 'img1', _) for _ in img_list if
                    ('jpg' in _) or ('png' in _)]

        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)
        self.tr_tracker = MOTR()

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.save_path = os.path.join(self.args.output_dir, 'results/{}'.format(seq_num))
        os.makedirs(self.save_path, exist_ok=True)

        self.predict_path = os.path.join(self.args.output_dir, 'preds', self.seq_num)
        os.makedirs(self.predict_path, exist_ok=True)
        if os.path.exists(os.path.join(self.predict_path, 'gt.txt')):
            os.remove(os.path.join(self.predict_path, 'gt.txt'))

    def load_img_from_file(self, f_path):
        """Load image and convert to proper format"""
        # For SportsMOT, we don't have labels_with_ids, so we just load the image
        cur_img = cv2.imread(f_path)
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        return cur_img, None

    def init_img(self, img):
        """Initialize image for processing"""
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    @staticmethod
    def write_results(txt_path, frame_id, bbox_xyxy, identities):
        """Write results in MOT format"""
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        with open(txt_path, 'a') as f:
            for xyxy, track_id in zip(bbox_xyxy, identities):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                line = save_format.format(frame=int(frame_id), id=int(track_id), x1=x1, y1=y1, w=w, h=h)
                f.write(line)

    def eval_seq(self):
        """Evaluate sequence using SportsMOT metrics"""
        data_root = os.path.join(self.args.mot_path, 'dataset/train')
        result_filename = os.path.join(self.predict_path, 'gt.txt')
        evaluator = SportsMOTEvaluator(data_root, self.seq_num)
        accs = evaluator.eval_file(result_filename)
        return accs

    @staticmethod
    def visualize_img_with_bbox(img_path, img, dt_instances: Instances, ref_pts=None, gt_boxes=None):
        """Visualize detection results"""
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if dt_instances.has('scores'):
            img_show = draw_bboxes(img, np.concatenate([dt_instances.boxes, dt_instances.scores.reshape(-1, 1)], axis=-1), dt_instances.obj_idxes)
        else:
            img_show = draw_bboxes(img, dt_instances.boxes, dt_instances.obj_idxes)
        if ref_pts is not None:
            img_show = draw_points(img_show, ref_pts)
        if gt_boxes is not None:
            img_show = draw_bboxes(img_show, gt_boxes, identities=np.ones((len(gt_boxes), )) * -1)
        cv2.imwrite(img_path, img_show)

    def detect(self, prob_threshold=0.7, area_threshold=100, vis=False):
        """Run detection and tracking on sequence"""
        total_dts = 0
        track_instances = None
        max_id = 0
        for i in tqdm(range(0, self.img_len)):
            img, targets = self.load_img_from_file(self.img_list[i])
            cur_img, ori_img = self.init_img(img)

            if track_instances is not None:
                track_instances.remove('boxes')
                track_instances.remove('labels')

            with torch.no_grad():
                res = self.detr.inference_single_image(cur_img.cuda().float(), (self.seq_h, self.seq_w), track_instances)
                track_instances = res['track_instances']
                max_id = max(max_id, track_instances.obj_idxes.max().item()) if len(track_instances) > 0 else max_id

                all_ref_pts = tensor_to_numpy(res['ref_pts'][0, :, :2])
                dt_instances = track_instances.to(torch.device('cpu'))

                # filter det instances by score and area
                dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
                dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

                total_dts += len(dt_instances)

                if vis:
                    # for visual
                    cur_vis_img_path = os.path.join(self.save_path, 'frame_{:06d}.jpg'.format(i))
                    gt_boxes = None
                    self.visualize_img_with_bbox(cur_vis_img_path, ori_img, dt_instances, ref_pts=all_ref_pts, gt_boxes=gt_boxes)

                tracker_outputs = self.tr_tracker.update(dt_instances)
                self.write_results(txt_path=os.path.join(self.predict_path, 'gt.txt'),
                                frame_id=(i + 1),
                                bbox_xyxy=tracker_outputs[:, :4],
                                identities=tracker_outputs[:, 5])
        print("totally {} dts max_id={}".format(total_dts, max_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and weights
    detr, _, _ = build_model(args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    detr = load_model(detr, args.resume)
    detr = detr.cuda()
    detr.eval()

    # SportsMOT sequences
    seq_nums = ['v_-6Os86HzwCs_c009', 'v_ApPxnw_Jffg_c016', 'v_gQNyhv8y0QY_c013']
    accs = []
    seqs = []

    for seq_num in seq_nums:
        print("Evaluating sequence: {}".format(seq_num))
        det = Detector(args, model=detr, seq_num=seq_num)
        det.detect(vis=True)
        accs.append(det.eval_seq())
        seqs.append(seq_num)

    # Standard MOT metrics
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    
    # Add HOTA metrics
    summary = hota_metrics.add_hota_to_summary(summary, accs, seqs)
    
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    
    # Save results to file
    with open(os.path.join(args.output_dir, "sportsmot_eval_results.txt"), 'w') as f:
        print(strsummary, file=f)
