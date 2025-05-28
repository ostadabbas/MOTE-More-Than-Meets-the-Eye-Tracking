# ------------------------------------------------------------------------
# Modified and add the copyrights as well to Institution (Author)
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import os
import numpy as np
import copy
import motmetrics as mm
mm.lap.default_solver = 'lap'
import os
from typing import Dict
import numpy as np
import logging

def read_results(filename, data_type: str, is_gt=False, is_ignore=False):
    if data_type in ('mot', 'lab'):
        read_fun = read_mot_results
    else:
        raise ValueError('Unknown data type: {}'.format(data_type))

    return read_fun(filename, is_gt, is_ignore)

def read_mot_results(filename, is_gt, is_ignore):
    valid_labels = {1}
    ignore_labels = {0, 2, 7, 8, 12}
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                results_dict.setdefault(fid, list())

                if is_gt:
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))
                        mark = int(float(linelist[6]))
                        if mark == 0 or label not in valid_labels:
                            continue
                    score = 1
                elif is_ignore:
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))
                        vis_ratio = float(linelist[8])
                        if label not in ignore_labels and vis_ratio >= 0:
                            continue
                    elif 'MOT15' in filename:
                        label = int(float(linelist[6]))
                        if label not in ignore_labels:
                            continue
                    else:
                        continue
                    score = 1
                else:
                    score = float(linelist[6])

                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])

                results_dict[fid].append((tlwh, target_id, score))

    return results_dict

def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)
    return tlwhs, ids, scores

class Evaluator(object):
    def __init__(self, data_root, seq_name, data_type='mot'):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type

        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):
        assert self.data_type == 'mot'

        gt_filename = os.path.join(self.data_root, self.seq_name, 'gt', 'gt.txt')
        self.gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True)
        self.gt_ignore_frame_dict = read_results(gt_filename, self.data_type, is_ignore=True)

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

        # ignore boxes
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]
        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

    def eval_file(self, filename):
        self.reset_accumulator()

        result_frame_dict = read_results(filename, self.data_type, is_gt=False)
        frames = sorted(list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)

        return self.acc

    def calculate_vote_metric(self, weights=(0.4, 0.4, 0.2)):
        total_subjects = len({obj[1] for frame_objs in self.gt_frame_dict.values() for obj in frame_objs})
        total_time_frames = max(self.gt_frame_dict.keys())

        # Initialize counters
        total_occlusions = 0
        total_occlusion_duration = 0

        visibility_consistency = 0
        occlusion_handling_efficiency = 0

        for subject_id in {obj[1] for frame_objs in self.gt_frame_dict.values() for obj in frame_objs}:
            occlusion_duration = 0
            visible_frames = 0
            occluded_frames = 0
            occlusion_started = False

            for frame_id in range(1, total_time_frames + 1):
                if subject_id in [obj[1] for obj in self.gt_frame_dict.get(frame_id, [])]:
                    if occlusion_started:
                        total_occlusion_duration += occlusion_duration
                        occlusion_duration = 0
                        occlusion_started = False
                    visible_frames += 1
                else:
                    occluded_frames += 1
                    occlusion_duration += 1
                    occlusion_started = True

            if visible_frames > 0:
                visibility_consistency += (visible_frames / total_time_frames)
            if occluded_frames > 0:
                total_occlusions += occluded_frames
                occlusion_handling_efficiency += (visible_frames / (visible_frames + occluded_frames))

        visibility_consistency /= total_subjects
        occlusion_handling_efficiency /= total_subjects
        average_occlusion_duration = total_occlusion_duration / total_occlusions if total_occlusions > 0 else 0

        metrics = {
            "Visibility Consistency": visibility_consistency,
            "Occlusion Handling Efficiency": occlusion_handling_efficiency,
            "Average Occlusion Duration": average_occlusion_duration
        }

        vc = metrics["Visibility Consistency"]
        ohe = metrics["Occlusion Handling Efficiency"]
        aod = metrics["Average Occlusion Duration"]

        # Ensure AOD is scaled
        max_aod = total_time_frames  # Use the total time frames as max possible occlusion duration
        aod_normalized = 1 - (aod / max_aod) if max_aod > 0 else 0

        w_vc, w_ohe, w_aod = weights

        vote = (w_vc * vc) + (w_ohe * ohe) + (w_aod * aod_normalized)

        return vote, metrics

    @staticmethod
    def get_summary(accs, names, metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()
