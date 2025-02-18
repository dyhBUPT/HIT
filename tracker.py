import numpy as np
from copy import copy
from os.path import join
from scipy.special import softmax
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

from track import BaseTrack, Track
from match import *


class Tracker:
    INFINITY = 1e5
    def __init__(
            self,
            dataset,
            video,
            detections,
            high_conf_thr,
            motion_param,
            iou_param,
            predictor,
            interpolation,
        ):
        BaseTrack.clear_count()

        self.dataset = dataset
        self.video = video
        self.predictor = predictor
        self.interpolation = interpolation
        self.delta_t = None
        self.high_conf_thr = high_conf_thr
        self.motion_param = motion_param
        self.iou_param = iou_param
        self.delta_xy = None
        self.allow_overlapping = False

        if isinstance(detections, str):
            self.load_tracks(detections)
        else:
            bboxes = detections.detections
            self.tracks = [
                Track(bbox[np.newaxis, :], predictor, interpolation) for bbox in bboxes
            ]
            self.frames = detections.frames

    def before_match(self, delta_t, curr_frame=None):
        self.delta_t = delta_t
        # initialize the matching dict
        self.matches = {
            idx: dict(matched=-1, matches=[])
            for idx in range(len(self.tracks))
        }
        # predict
        for track in self.tracks:
            # By specifying `curr_frame`, only nearby tracks will perform prediction.
            if curr_frame is not None:
                if (0 <= track.f_min - curr_frame <= self.delta_t) or \
                   (0 <= curr_frame - track.f_max <= self.delta_t):
                    track.predict(self.delta_t, conf_thr=self.high_conf_thr, delta_xy=self.delta_xy)
            else:
                track.predict(self.delta_t, conf_thr=self.high_conf_thr, delta_xy=self.delta_xy)

    def camera_motion_compensation(self, **cmc_param):
        if not cmc_param['enable']:
            return

        assert self.delta_t == 1
        cmc_iou_thr, cmc_tau = cmc_param['thr'], cmc_param['tau']
        iou_thr = self.iou_param['thr']

        def gaussian_smooth(inputs, tau):
            if tau is None:
                return inputs
            inputs = np.array(inputs).reshape(-1, 1)
            gpr = GPR(RBF(tau, 'fixed'))
            t = np.arange(len(inputs)).reshape(-1, 1)
            gpr.fit(t, inputs)
            outputs = gpr.predict(t).tolist()
            return outputs

        # compute the average inter-frame IoU / movement
        IOUS, DELTA_XY = [], [[0, 0]]
        for frame in range(self.frames[0], self.frames[1]):
            # iou-based matching for high-score bboxes
            self.compute_cost(frame, iou_thr=1-iou_thr)
            mask_row = np.array([
                i for i, idx in enumerate(self.index_i)
                if self.tracks[idx].score >= self.high_conf_thr
            ], dtype=int)
            mask_col = np.array([
                j for j, idx in enumerate(self.index_j)
                if self.tracks[idx].score >= self.high_conf_thr
            ], dtype=int)[:, np.newaxis]
            mask = np.ones_like(self.iou_cost_matrix) * self.INFINITY
            mask[mask_row, mask_col] = 0
            cost_matrix = np.maximum(self.iou_cost_matrix, mask)
            matches, un_match_i, un_match_j = linear_assignment(cost_matrix, 1 - iou_thr)
            un_match_i = [i for i in un_match_i if self.tracks[self.index_i[i]].score >= self.high_conf_thr]
            un_match_j = [j for j in un_match_j if self.tracks[self.index_j[j]].score >= self.high_conf_thr]

            # gathering the inter-frame IoUs / movement
            ious, delta_xy = [], []
            xy_i_, xy_j_ = [], []
            for i, j in matches:
                ious.append(1 - cost_matrix[i, j])
                det_i = self.tracks[self.index_i[i]].dets[frame]
                det_j = self.tracks[self.index_j[j]].dets[frame + 1]
                xy_i = det_i[:2] + det_i[2:4] / 2
                xy_j = det_j[:2] + det_j[2:4] / 2
                delta_xy.append(xy_j - xy_i)
                xy_i_.append(xy_i)
                xy_j_.append(xy_j)
            ious += [0] * (len(un_match_i) + len(un_match_j))
            if len(ious) > 0:
                IOUS.append(np.mean(ious))
            if len(delta_xy) > 0:
                DELTA_XY.append(np.mean(delta_xy, axis=0).tolist())
            else:
                DELTA_XY.append([0, 0])
        DELTA_XY = np.array(DELTA_XY)
        IOU = np.mean(IOUS)

        # compute the inter-frame compensation
        # Please note that we use the `sequence-level` average IoU to recognize camera movements.
        # For better performance, you can try to design the `clip-level` version.
        if IOU < cmc_iou_thr:
            delta_x = gaussian_smooth(DELTA_XY[:, 0], cmc_tau)
            delta_y = gaussian_smooth(DELTA_XY[:, 1], cmc_tau)
            delta_x = np.cumsum(delta_x)
            delta_y = np.cumsum(delta_y)
            self.delta_xy = {
                f: np.array([delta_x[i], delta_y[i]])
                for i, f in enumerate(range(self.frames[0], self.frames[1] + 1))
            }
            # redo the tracklet prediction
            self.before_match(self.delta_t)

    def compute_cost(self, curr_frame, motion_thr=None, iou_thr=None, cls=False):
        self.index_i = [
            i for i, trk in enumerate(self.tracks)
            if (trk.f_max == curr_frame) and
               (self.delta_t == 1 or trk.score_max >= self.high_conf_thr)
        ]

        if self.allow_overlapping:
            # allow temporally overlapping tracklets to be matched based on motion costs
            self.index_j = [
                j for j, trk in enumerate(self.tracks)
                if (-self.delta_t <= trk.f_min - curr_frame <= self.delta_t) and
                   (self.delta_t == 1 or trk.score_max >= self.high_conf_thr)
            ]
            max_len = self.delta_t
            iou_thr = None
        else:
            self.index_j = [
                j for j, trk in enumerate(self.tracks)
                if (1 <= trk.f_min - curr_frame <= self.delta_t) and
                   (self.delta_t == 1 or trk.score_max >= self.high_conf_thr)
            ]
            max_len = 1

        # class mask for multi-class tracking [`hard` mode]
        if cls:
            cls_i = np.array([self.tracks[i].cid for i in self.index_i])
            cls_j = np.array([self.tracks[j].cid for j in self.index_j])
            cls_mask = cls_i.reshape(-1, 1) != cls_j
        else:
            cls_mask = np.zeros([len(self.index_i), len(self.index_j)], dtype=bool)

        # compute motion cost
        if motion_thr is not None:
            self.motion_cost_matrix = np.ones([len(self.index_i), len(self.index_j)])
            for i, idx_i in enumerate(self.index_i):
                for j, idx_j in enumerate(self.index_j):
                    if idx_i == idx_j:
                        continue
                    self.motion_cost_matrix[i, j] = multi_frame_motion_cost(
                        self.tracks[idx_i],
                        self.tracks[idx_j],
                        self.motion_param,
                        max_len=max_len,
                    )
            self.motion_cost_matrix[self.motion_cost_matrix > motion_thr] = self.INFINITY
            self.motion_cost_matrix[cls_mask] = self.INFINITY
        else:
            self.motion_cost_matrix = None

        # compute iou cost
        if iou_thr is not None:
            self.iou_cost_matrix = np.ones([len(self.index_i), len(self.index_j)])
            for i, idx_i in enumerate(self.index_i):
                for j, idx_j in enumerate(self.index_j):
                    if idx_i == idx_j:
                        continue
                    self.iou_cost_matrix[i, j] = multi_frame_iou_cost(
                        self.tracks[idx_i],
                        self.tracks[idx_j],
                        self.iou_param,
                        self.delta_xy,
                    )
            self.iou_cost_matrix[self.iou_cost_matrix > iou_thr] = self.INFINITY
            self.iou_cost_matrix[cls_mask] = self.INFINITY
        else:
            self.iou_cost_matrix = None

    def pre_match(self, enable):
        if not enable:
            return

        for frame in range(self.frames[0], self.frames[1]):
            self.match(frame)
        self.update()

        new_tracks = []
        for track in self.tracks:
            for frame, det in track.dets.items():
                # We don't use `CMC` and `Motion` simultaneously to avoid inaccurate prediction here.
                # Looking forward to your better solutions!
                if self.delta_xy is None:
                    dets = track.dets
                else:
                    dets = {frame: det}
                # equip detections with motion information
                pred_dets = dict()
                pred_dets.update(
                    track.motion_predictor.predict(
                        dets,
                        f_start=frame - self.delta_t,
                        f_stop=frame - 1,
                        f_curr=frame,
                        conf_thr=self.high_conf_thr,
                        delta_xy=self.delta_xy,
                    )
                )
                pred_dets.update(
                    track.motion_predictor.predict(
                        dets,
                        f_start=frame + 1,
                        f_stop=frame + self.delta_t,
                        f_curr=frame,
                        conf_thr=self.high_conf_thr,
                        delta_xy=self.delta_xy,
                    )
                )
                new_det = np.array([frame] + det.tolist() + [track.cid])
                new_track = Track(new_det[np.newaxis, :], track.predictor, track.interpolation)
                new_track.pred_dets = pred_dets
                new_tracks.append(new_track)

        self.tracks = new_tracks

        self.matches = {
            idx: dict(matched=-1, matches=[])
            for idx in range(len(self.tracks))
        }

    def match(self, frame, BYTE=True):
        motion_thr = self.motion_param['thr']
        iou_thr = self.iou_param['thr']

        self.compute_cost(frame, motion_thr=1-motion_thr, iou_thr=1-iou_thr)

        # add constraints for long-term association on KITTI
        if (self.dataset == 'KITTI') and (self.delta_t >= 10):
            motion_thr += .6
            iou_thr += .6

        # use `BYTE` only for consecutive frames
        if self.delta_t == 1:

            '''first-stage matching -> high-score & motion-based'''
            mask_first_row = np.array([
                i for i, idx in enumerate(self.index_i)
                if self.tracks[idx].score >= self.high_conf_thr
            ], dtype=int)
            mask_first_col = np.array([
                j for j, idx in enumerate(self.index_j)
                if self.tracks[idx].score >= self.high_conf_thr
            ], dtype=int)[:, np.newaxis]
            mask_first = np.ones_like(self.motion_cost_matrix) * self.INFINITY
            mask_first[mask_first_row, mask_first_col] = 0
            cost_matrix_first = np.maximum(self.motion_cost_matrix, mask_first)
            matches_first, un_match_i_first, un_match_j_first = linear_assignment(cost_matrix_first, 1 - motion_thr)

            '''second-stage matching -> low-score & IoU-based'''
            # Note that the matching between two low-score tracklets are allowed here.
            if BYTE and (self.iou_cost_matrix is not None):
                mask_second_row = np.array([
                    i for i, idx in enumerate(self.index_i)
                    if (self.tracks[idx].score < self.high_conf_thr) or (i in un_match_i_first)
                ], dtype=int)
                mask_second_col = np.array([
                    j for j, idx in enumerate(self.index_j)
                    if (self.tracks[idx].score < self.high_conf_thr) or (j in un_match_j_first)
                ], dtype=int)[:, np.newaxis]
                mask_second = np.ones_like(self.iou_cost_matrix) * self.INFINITY
                mask_second[mask_second_row, mask_second_col] = 0
                cost_matrix_second = np.maximum(self.iou_cost_matrix, mask_second)
                matches_second, un_match_i_second, un_match_j_second = linear_assignment(cost_matrix_second, 1 - iou_thr)
            else:
                matches_second = np.array([])

        else:

            '''first-stage matching -> high-score & motion-based'''
            # Note that we use `score_max` instead of `score` to select high-score tracklets here
            mask_first_row = np.array([
                i for i, idx in enumerate(self.index_i)
                if (self.tracks[idx].score_max >= self.high_conf_thr)
            ], dtype=int)
            mask_first_col = np.array([
                j for j, idx in enumerate(self.index_j)
                if (self.tracks[idx].score_max >= self.high_conf_thr)
            ], dtype=int)[:, np.newaxis]
            mask_first = np.ones_like(self.motion_cost_matrix) * self.INFINITY
            mask_first[mask_first_row, mask_first_col] = 0
            cost_matrix_first = np.maximum(self.motion_cost_matrix, mask_first)
            matches_first, un_match_i_first, un_match_j_first = linear_assignment(cost_matrix_first, 1 - motion_thr)

            '''second-stage matching -> high-score & IoU-based'''
            if (self.iou_cost_matrix is not None) and (un_match_i_first.size * un_match_j_first.size > 0):
                mask_second_row = np.array([
                    i for i, idx in enumerate(self.index_i)
                    if (self.tracks[idx].score >= self.high_conf_thr) and (i in un_match_i_first)
                ], dtype=int)
                mask_second_col = np.array([
                    j for j, idx in enumerate(self.index_j)
                    if (self.tracks[idx].score >= self.high_conf_thr) and (j in un_match_j_first)
                ], dtype=int)[:, np.newaxis]
                mask_second = np.ones_like(self.iou_cost_matrix) * self.INFINITY
                mask_second[mask_second_row, mask_second_col] = 0
                cost_matrix_second = np.maximum(self.iou_cost_matrix, mask_second)
                matches_second, un_match_i_second, un_match_j_second = linear_assignment(cost_matrix_second, 1 - iou_thr)
            else:
                matches_second = np.array([])

        if (matches_first.size > 0) and (matches_second.size > 0):
            matches = np.concatenate([matches_first, matches_second], axis=0)
        elif matches_first.size > 0:
            matches = matches_first
        else:
            matches = matches_second

        '''update the matching dict'''
        for match_i, match_j in matches:
            index_i = self.index_i[match_i]
            index_j = self.index_j[match_j]
            matched_i = self.matches[index_i]['matched']
            matched_j = self.matches[index_j]['matched']
            matches_j = self.matches[index_j]['matches']
            new_matches = [index_j] + matches_j
            if matched_j != -1:  # trk_j has been matched by another trk
                continue
            if matched_i == -1:
                self.matches[index_i]['matches'].extend(new_matches)
                self.matches[index_j]['matched'] = index_i
            else:
                self.matches[matched_i]['matches'].extend(new_matches)
                self.matches[index_j]['matched'] = matched_i

    def update(self):
        reserved_tracks = []
        for track_idx, info in self.matches.items():
            if info['matched'] != -1:
                continue
            reserved_tracks.append(track_idx)
            track = self.tracks[track_idx]
            for idx in info['matches']:
                track.update(self.tracks[idx])
        self.tracks = [self.tracks[idx] for idx in reserved_tracks]

    def postprocess(self, min_len=None, merge=None, smooth=False):
        # remove short tracklets
        if min_len is not None:
            self.tracks = [
                track for track in self.tracks
                if len(track.dets) > min_len
            ]

        # merge overlapping tracklets
        if merge is not None:
            cost_matrix = tracklet_cost(self.tracks, merge)
            self.tracks = merging_tracklets(self.tracks, cost_matrix, merge['thr'])

        # remove low-score tracklets
        self.tracks = [
            track for track in self.tracks
            if track.score >= self.high_conf_thr
        ]

        # smooth
        if smooth:
            for track in self.tracks:
                track.smooth(conf=self.high_conf_thr)

    def write(self, output_dir, min_area=-1, max_ratio=1e5, fmt='MOT17'):
        file_name = join(output_dir, f'{self.video}.txt')
        with open(file_name, 'w') as f:
            for track in self.tracks:
                for fid, bbox in track.dets.items():
                    if bbox[2] * bbox[3] < min_area:
                        continue
                    if bbox[2] / bbox[3] > max_ratio:
                        continue
                    if fmt == 'KITTI':
                        classes = {0: 'Pedestrian', 1: 'Car'}
                        f.write(
                            '{:.0f} {:.0f} {} -1 -1 -1 {:.2f} {:.2f} {:.2f} {:.2f}'
                            ' -1 -1 -1 -1000 -1000 -1000 -10 {:.2f}\n'
                            .format(
                                fid, track.track_id, classes[track.cid],
                                bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[4]
                            )
                        )
                    else:
                        f.write(
                            '%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,-1,-1,%d\n' % (
                                fid, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5],
                            )
                        )
        if self.dataset == 'DanceTrack':
            extra_association(file_name)

    def __repr__(self):
        return f'Tracker_{self.video}_{len(self.tracks)}'

    def load_tracks(self, path, interval=1):
        self.tracks = []
        trks = np.loadtxt(path, delimiter=',')
        trks = trks[np.lexsort([trks[:, 0], trks[:, 1]])]  # ID->frame
        self.frames = (int(np.min(trks[:, 0])), int(np.max(trks[:, 0])))
        ids = set(trks[:, 1])
        for oid in ids:
            trk_id = trks[trks[:, 1] == oid]
            curr_trk = []
            prev_frame = trk_id[0, 0] - 1
            for i, row in enumerate(trk_id):
                curr_frame = row[0]
                if (curr_frame > prev_frame + interval) or ():
                    if len(curr_trk) > 0:
                        self.tracks.append(
                            Track(
                                np.array(curr_trk),
                                self.predictor,
                                self.interpolation,
                            )
                        )
                        curr_trk = []
                curr_trk.append(
                    np.delete(row, [1, 7, 8]).tolist()
                )
                prev_frame = curr_frame
            # store the last tracklets
            if len(curr_trk) > 0:
                self.tracks.append(
                    Track(
                        np.array(curr_trk),
                        self.predictor,
                        self.interpolation,
                    )
                )
