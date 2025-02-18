import lap
import numpy as np
from copy import copy
from collections import defaultdict


def linear_assignment(cost_matrix, thresh=np.inf, mode='early_thresh'):
    """
    Note:
    - By using `early_thresh`, the maximum cost is limited to `thresh` as in previous works (e.g., ByteTrack).
    - By using `late_thresh`, unlimited cost matrix is applied, and the matches are further revised with `thresh`.
    """
    assert mode in ('early_thresh', 'late_thresh')
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.array(range(cost_matrix.shape[0])),
            np.array(range(cost_matrix.shape[1])),
        )
    matches, unmatched_a, unmatched_b = [], [], []

    if mode == 'early_thresh':
        cost_limit = thresh
    elif mode == 'late_thresh':
        cost_limit = np.inf

    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=cost_limit)

    unmatched_a = np.where(x < 0)[0].tolist()
    unmatched_b = np.where(y < 0)[0].tolist()
    for ix, mx in enumerate(x):
        if mx >= 0:
            if cost_matrix[ix, mx] <= thresh:
                matches.append([ix, mx])
            else:  # to revise the matching results for the `late_thresh` mode.
                unmatched_a.append(ix)
                unmatched_b.append(mx)

    return np.asarray(matches), np.asarray(unmatched_a), np.asarray(unmatched_b)


def compute_iou(bbox1, bbox2, **params):
    """
    bbox: [x,y,w,h,s]
    """
    modes = params['mode'].split('_')
    if 'cons' in modes:
        W = params['cons_w']
        S = params['cons_score']
        w1, w2 = bbox1[2], bbox2[2]
        s1, s2 = bbox1[4], bbox2[4]
        if (w1 < W and w2 < W) and (s1 > S or s2 > S):
            bbox1 = copy(bbox1)
            bbox2 = copy(bbox2)
            center1 = bbox1[:2] + bbox1[2:4] / 2
            center2 = bbox2[:2] + bbox2[2:4] / 2
            ratio_1 = np.exp(W / max(1, w1) * params['cons_tau'])
            ratio_2 = np.exp(W / max(1, w2) * params['cons_tau'])
            ratio = np.sqrt(ratio_1 * ratio_2)
            bbox1[2:4] *= ratio
            bbox2[2:4] *= ratio
            bbox1[:2] = center1 - bbox1[2:4] / 2
            bbox2[:2] = center2 - bbox2[2:4] / 2
    # compute iou
    x1, y1, w1, h1 = bbox1[:4]
    x2, y2, w2, h2 = bbox2[:4]
    area1 = w1 * h1
    area2 = w2 * h2
    X1 = max(x1, x2)
    Y1 = max(y1, y2)
    X2 = min(x1 + w1, x2 + w2)
    Y2 = min(y1 + h1, y2 + h2)
    W = max(0, X2 - X1)
    H = max(0, Y2 - Y1)
    inter = W * H
    if 'iou' in modes:
        iou = inter / (area1 + area2 - inter)
    elif 'miou' in modes:
        iou = inter / min(area1, area2)
    else:
        raise RuntimeError('error iou mode')
    if 'hm' in modes:
        iou_h = H / (h1 + h2 - H)
        iou *= iou_h
    return iou


def motion_cost(trk_i, trk_j, motion_param):
    assert trk_i.f_max < trk_j.f_min
    bbox_i = trk_i.dets[trk_i.f_max]
    bbox_i_pred = trk_i.pred_dets[trk_j.f_min]
    bbox_j = trk_j.dets[trk_j.f_min]
    bbox_j_pred = trk_j.pred_dets[trk_i.f_max]
    iou_i2j = compute_iou(bbox_i, bbox_j_pred, **motion_param)
    iou_j2i = compute_iou(bbox_j, bbox_i_pred, **motion_param)
    return 1 - (iou_i2j + iou_j2i) / 2


def multi_frame_motion_cost(trk_i, trk_j, motion_param, max_len=5):
    """
    max_len: the maximum length for cost calculation.
    """
    IOUs = []
    # IoU between predicted dets of trk_i and true dets of trk_j
    for f, pred_det in trk_i.pred_dets.items():
        if f - trk_j.f_min >= max_len:
            continue
        det = trk_j.dets.get(f)
        if det is not None:
            IOUs.append(compute_iou(pred_det, det, **motion_param))
    # IoU between predicted dets of trk_j and true dets of trk_i
    for f, pred_det in trk_j.pred_dets.items():
        if trk_i.f_max - f >= max_len:
            continue
        det = trk_i.dets.get(f)
        if det is not None:
            IOUs.append(compute_iou(pred_det, det, **motion_param))
    # return motion cost
    if len(IOUs) == 0:
        return 1
    else:
        return 1 - np.mean(IOUs)


def multi_frame_iou_cost(trk_i, trk_j, iou_param, delta_xy=None):
    # temporally non-overlapping trks
    if trk_i.f_max < trk_j.f_min:
        det_i = trk_i.dets[trk_i.f_max]
        det_j = trk_j.dets[trk_j.f_min]
        if delta_xy is not None:
            det_i = copy(det_i)
            det_i[:2] += delta_xy[trk_j.f_min] - delta_xy[trk_i.f_max]
        iou = compute_iou(det_i, det_j, **iou_param)
    # temporally overlapping trks
    else:
        # IoU in overlapping frames
        IOUs = []
        for f in range(trk_j.f_min, trk_i.f_max + 1):
            det_i = trk_i.dets.get(f)
            det_j = trk_j.dets.get(f)
            if (det_i is not None) and (det_j is not None):
                IOUs.append(compute_iou(det_i, det_j, **iou_param))
        if len(IOUs) == 0:
            iou = 0
        else:
            iou = np.mean(IOUs)

    return 1 - iou


def tracklet_cost(tracks, param):
    cost_matrix = np.ones([len(tracks), len(tracks)])
    # compute cost between trk_i and its subsequent trk_j
    for i, track_i in enumerate(tracks):
        for j, track_j in enumerate(tracks):
            if i == j:
                continue
            if not (track_i.f_min <= track_j.f_min <= track_i.f_max):
                continue
            ious = []
            for f in range(track_j.f_min, min(track_i.f_max, track_j.f_max) + 1):
                det_i = track_i.dets.get(f)
                det_j = track_j.dets.get(f)
                if (det_i is None) or (det_j is None):
                    continue
                iou = compute_iou(det_i, det_j, **param)
                ious.append(iou)
            cost_matrix[i, j] = 1 - sum(ious) / max(1e-5, len(ious))
    return cost_matrix


def merging_tracklets(tracks, cost_matrix, iou_thr, mode='loose'):
    matches = {
        idx: dict(matched=-1, matches=[])
        for idx in range(len(tracks))
    }
    assert mode in ('strict', 'loose')

    for index_i in range(len(tracks)):

        # one trk_i can only be matched with one subsequent trk_j (with the minimum cost)
        if mode == 'strict':
            # search for the best match
            matches_i = None
            for index_j in range(len(tracks)):
                matched_j = matches[index_j]['matched']
                # trk_j has high cost with trk_i
                if cost_matrix[index_i, index_j] > 1 - iou_thr:
                    continue
                # trk_j has been matched by another trk
                if matched_j != -1:
                    continue
                # update the matches
                if (matches_i is None) or \
                   (cost_matrix[index_i, index_j] < cost_matrix[index_i, matches_i]):
                    matches_i = index_j
            # update the matching dict
            if matches_i is not None:
                matched_idx = matches[index_i]['matched']
                new_matches = [matches_i] + matches[matches_i]['matches']
                if matched_idx == matches_i:
                    continue
                elif matched_idx == -1:
                    matches[index_i]['matches'].extend(new_matches)
                    matches[matches_i]['matched'] = index_i
                else:
                    matches[matched_idx]['matches'].extend(new_matches)
                    matches[matches_i]['matched'] = matched_idx

        # one trk_i can be matched with multiple subsequent trk_j
        elif mode == 'loose':
            for index_j in range(len(tracks)):
                matched_i = matches[index_i]['matched']
                matches_i = matches[index_i]['matches']
                matched_j = matches[index_j]['matched']
                # trk_j has high cost with trk_i
                if cost_matrix[index_i, index_j] > 1 - iou_thr:
                    continue
                # trk_i has been matched to trk_j
                if matched_i == index_j:
                    continue
                # trk_j has been matched by another trk
                if matched_j != -1:
                    continue
                # if trk_j has high cost with an matched and temporally overlapping trk_mi, ignore trk_j
                if len(matches_i) > 0:
                    FLAG = False
                    for mi in matches_i:
                        trk_mi = tracks[mi]
                        trk_j = tracks[index_j]
                        if (trk_mi.f_max < trk_j.f_min) or (trk_mi.f_min > trk_j.f_max):
                            continue
                        if min(cost_matrix[mi, index_j], cost_matrix[index_j, mi]) > 1 - iou_thr:
                            FLAG = True
                    if FLAG:
                        continue
                # match trk_j (and its matched trks) to trk_i
                new_matches = [index_j] + matches[index_j]['matches']
                if matched_i == -1:
                    # matches[index_i]['matches'].append(index_j)
                    matches[index_i]['matches'].extend(new_matches)
                    matches[index_j]['matched'] = index_i
                else:
                    # matches[matched_i]['matches'].append(index_j)
                    matches[matched_i]['matches'].extend(new_matches)
                    matches[index_j]['matched'] = matched_i

    reserved_tracks = []
    for track_idx, info in matches.items():
        if info['matched'] != -1:
            continue
        reserved_tracks.append(track_idx)
        track = tracks[track_idx]
        matches_idx = sorted(set(info['matches']), key=lambda idx: tracks[idx].f_min)
        for idx in matches_idx:
            track.update(tracks[idx])
    tracks = [tracks[idx] for idx in reserved_tracks]
    return tracks


class FindUnionSet(dict):
    def find(self, src):
        if src in self:
            return self.find(self[src])
        return src

    def merge(self, dst, src):
        self[self.find(src)] = self.find(dst)


def extra_association(file, t_min=20, t_max=100):
    """
    This is for extra association on DanceTrack.
    ref: https://github.com/megvii-research/MOTRv2/blob/main/tools/merge_dance_tracklets.py
    """
    with open(file) as f:
        lines = f.readlines()
    instance_timestamps = defaultdict(list)
    for line in lines:
        f_id, pid = map(int, line.split(',')[:2])
        instance_timestamps[pid].append(f_id)
    instances = list(instance_timestamps.keys())
    fid_map = FindUnionSet()
    for i in instances:
        for j in instances:
            if fid_map.find(i) == fid_map.find(j):
                continue
            end_t = max(instance_timestamps[i])
            start_t = min(instance_timestamps[j])
            if sum([0 <= start_t - max(pts) < t_max for pts in instance_timestamps.values()]) > 1:
                continue
            if sum([0 <= min(pts) - end_t < t_max for pts in instance_timestamps.values()]) > 1:
                continue
            dt = start_t - end_t
            if t_min < dt < t_max:
                fid_map.merge(i, j)

    with open(file, 'w') as f:
        for line in lines:
            f_id, pid, *info = line.split(',')
            pid = str(fid_map.find(int(pid)))
            f.write(','.join([f_id, pid, *info]))
