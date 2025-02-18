"""
Hierarchical IoU Tracking based on temporal window
"""
from tqdm import tqdm
from os.path import join

from opts import opt
from tracker import *
from detection import Detection
from evaluation import evaluation


class Tracker_W(Tracker):
    def __init__(self, **kwargs):
        super(Tracker_W, self).__init__(**kwargs)

    def before_match(self, delta_t, max_interval):
        self.delta_t = delta_t
        # initialize the matching dict
        self.matches = {
            idx: dict(matched=-1, matches=[])
            for idx in range(len(self.tracks))
        }
        # predict
        pred_frames = min(delta_t, max_interval)
        for track in self.tracks:
            track.predict(pred_frames, conf_thr=self.high_conf_thr, delta_xy=self.delta_xy)

    def compute_cost(self, curr_frame, motion_thr, iou_thr, thr_t):
        self.index_i = [
            i for i, trk in enumerate(self.tracks)
            if (curr_frame <= trk.f_min <= trk.f_max < curr_frame + self.delta_t) and
               (self.delta_t <= thr_t or trk.score_max >= self.high_conf_thr)
        ]
        self.index_j = copy(self.index_i)
        assert not self.allow_overlapping
        max_len = 1

        # temporal mask
        FMAX = np.array([self.tracks[i].f_max for i in self.index_i])
        FMIN = np.array([self.tracks[j].f_min for j in self.index_j])
        temporal_mask = FMAX.reshape(-1, 1) >= FMIN

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
            self.motion_cost_matrix[temporal_mask] = self.INFINITY
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
            self.iou_cost_matrix[temporal_mask] = self.INFINITY
        else:
            self.iou_cost_matrix = None

    def match(self, frame, DELTA_T, BYTE=True):
        motion_thr = self.motion_param['thr']
        iou_thr = self.iou_param['thr']

        self.compute_cost(frame, motion_thr=1-motion_thr, iou_thr=1-iou_thr, thr_t=DELTA_T)

        if self.delta_t <= DELTA_T:

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


def run(args):
    for video in tqdm(args.videos):
        if args.params['input_tracks']:
            detections = join(args.input_dir, f'{video}.txt')
        else:
            detections = Detection(
                input_path=join(args.input_dir, f'{video}.txt'),
                conf_thr=args.params['conf_thr']['low'],
                nms_thr=args.params['nms_thr'],
            )

        tracker = Tracker_W(
            dataset=args.dataset,
            video=video,
            detections=detections,
            high_conf_thr=args.params['conf_thr']['high'],
            motion_param=args.params['motion_matching'],
            iou_param=args.params['iou_matching'],
            predictor=args.params['predictor'],
            interpolation=args.params['interpolation'],
        )

        delta_t = 1
        total_frames = tracker.frames[1] - tracker.frames[0] + 1
        while delta_t <= total_frames:
            delta_t *= 2
            tracker.before_match(delta_t, max_interval=60)
            for frame in range(tracker.frames[0], tracker.frames[1], delta_t):
                tracker.match(frame, DELTA_T=8)
            tracker.update()

        tracker.postprocess(**args.params['post_trk'])
        tracker.write(
            output_dir=args.output_dir,
            fmt=args.dataset,
            **args.params['post_det'],
        )

    if args.split == 'val':
        evaluation()


if __name__ == '__main__':
    run(opt)
