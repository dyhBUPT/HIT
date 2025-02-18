import numpy as np
from copy import copy
from filterpy.kalman import KalmanFilter


class LinearPredictor:
    def __init__(self, delta_t, preserve_scale):
        self.delta_t = delta_t
        self.preserve_wh = preserve_scale

    def predict(self, dets, f_start, f_stop=None, conf_thr=None, delta_xy=None, f_curr=None):
        if f_stop is None:
            f_stop = f_start
        if conf_thr is None:
            conf_thr = -1
        assert f_start <= f_stop

        pred_dets = dict()

        # setting the point for prediction
        if f_curr is not None:
            f_min, f_max = f_curr, f_curr
        else:
            f_min, f_max = min(dets), max(dets)

        if f_start <= f_stop < f_min:
            # the stop point for prediction
            f1 = f_min
            while dets.get(f1)[4] < conf_thr:
                if dets.get(f1 + 1) is not None:
                    f1 += 1
                else:
                    break
            # the start point for prediction
            f0 = f1
            for i in range(self.delta_t):
                dt = self.delta_t - i
                det = dets.get(f1 + dt)
                if det is not None:
                    if det[4] < conf_thr:
                        continue
                    f0 = f1 + dt
                    break
        elif f_max < f_start <= f_stop:
            # the stop point for prediction
            f1 = f_max
            while dets.get(f1)[4] < conf_thr:
                if dets.get(f1 - 1) is not None:
                    f1 -= 1
                else:
                    break
            # the start point for prediction
            f0 = f1
            for i in range(self.delta_t):
                dt = self.delta_t - i
                det = dets.get(f1 - dt)
                if det is not None:
                    if det[4] < conf_thr:
                        continue
                    f0 = f1 - dt
                    break
        else:
            raise RuntimeError('frame error')

        default = np.array([0, 0])
        det0 = dets[f0]  # the start point for prediction
        det1 = copy(dets[f1])  # the stop point for prediction

        if delta_xy is not None:
            det1[:2] += np.array(delta_xy.get(f1, default)) - \
                    np.array(delta_xy.get(f0, default))

        det_delta = (det1 - det0) / max(1, abs(f0 - f1))  # step for prediction
        for f in range(f_start, f_stop + 1):
            det_f = det1 + det_delta * abs(f - f1)

            if delta_xy is not None:
                det_f[:2] += delta_xy.get(f, default) - \
                             delta_xy.get(f1, default)

            if self.preserve_wh:
                pred_dets[f] = np.array(det_f[:2].tolist() + det1[2:].tolist())
            else:
                pred_dets[f] = np.array(det_f[:4].tolist() + det1[4:].tolist())
        return pred_dets


class KalmanPredictor:
    """
    ref: https://github.com/noahcao/OC_SORT/blob/master/trackers/ocsort_tracker/ocsort.py
    """
    def __init__(self, delta_t, preserve_scale):
        self.delta_t = delta_t
        self.preserve_scale = preserve_scale

    def predict(self, dets, f_start, f_stop=None, conf_thr=None, delta_xy=None, f_curr=None):
        if f_stop is None:
            f_stop = f_start
        if conf_thr is None:
            conf_thr = -1
        assert f_start <= f_stop

        pred_dets = dict()

        # setting the point for prediction
        if f_curr is not None:
            f_min, f_max = f_curr, f_curr
        else:
            f_min, f_max = min(dets), max(dets)

        # get the frame range `triplet` for prediction
        if f_start <= f_stop < f_min:
            f0 = f_min
            for i in range(self.delta_t):
                dt = self.delta_t - i
                if f_min + dt in dets:
                    f0 = f_min + dt
                    break
            triplet = (f0 - 1, f_start - 1, -1)
        elif f_max < f_start <= f_stop:
            f0 = f_max
            for i in range(self.delta_t):
                dt = self.delta_t - i
                if f_max - dt in dets:
                    f0 = f_max - dt
                    break
            triplet = (f0 + 1, f_stop + 1, 1)
        else:
            raise RuntimeError('frame error')

        kf = self.get_kf(dets[f0])
        default = np.array([0, 0])
        for f in range(*triplet):

            if delta_xy is not None:
                x = copy(kf.x)
                x[:2, 0] += np.array(delta_xy.get(f, default)) - \
                            np.array(delta_xy.get(f0, default))
                kf.x = x.reshape(-1, 1)

            pred_bbox = self.kf_predict(kf)

            if f_start <= f <= f_stop:
                pred_dets[f] = np.array(pred_bbox)

            bbox = dets.get(f)

            # As to the low-score observation, use predicted state instead.
            if (bbox is not None) and (bbox[4] < conf_thr):
                bbox = pred_bbox

            self.kf_update(kf, bbox)

        return pred_dets

    def convert_bbox_to_z(self, bbox):
        """
        Takes a bounding box in the form [x1,y1,w,h,score] and returns z in the form
        [x,y,s,score,r] where x,y is the centre of the box and s is the scale(area)
        and r is the aspect ratio
        """
        w, h = bbox[2:4]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h  # scale is just area
        r = w / float(h + 1e-6)
        score = bbox[4]
        return np.array([x, y, s, score, r]).reshape((5, 1))

    def convert_x_to_bbox(self, x):
        """
        Takes a bounding box in the centre form [x,y,s,score,r]
        and returns it in the form [x1,y1,w,h,score]
        """
        x, y, s, score, r = x[:5, 0]
        w = np.sqrt(max(1e-6, s * r))
        h = s / w
        return np.array([x - w / 2., y - h / 2., w, h, score])

    def get_kf(self, bbox):
        kf = KalmanFilter(dim_x=9, dim_z=5)
        kf.F = np.array([
            [1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ])

        kf.R[2:, 2:] *= 10.
        kf.P[5:, 5:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        kf.P *= 10.
        kf.Q[-1, -1] *= 0.01
        kf.Q[-2, -2] *= 0.01
        kf.Q[5:, 5:] *= 0.01

        kf.x[:5] = self.convert_bbox_to_z(bbox)
        return kf

    def kf_predict(self, kf):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (kf.x[6] + kf.x[2]) <= 0:
            kf.x[6] *= 0.0
        kf.predict()
        bbox = self.convert_x_to_bbox(kf.x)
        return bbox

    def kf_update(self, kf, bbox, nsa=False):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is None:
            # set `~s` to 0 when the observation is unavailable.
            if self.preserve_scale:
                kf.x[7] = 0
            kf.update(bbox)
        else:
            if nsa:
                R = kf.R * (1 - bbox[4])
            else:
                R = None
            kf.update(self.convert_bbox_to_z(bbox), R=R)
