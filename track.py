import numpy as np
from copy import copy

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

from opts import opt
from predictor import LinearPredictor, KalmanPredictor


class BaseTrack:
    id_count = 0

    @staticmethod
    def next_id():
        BaseTrack.id_count += 1
        return BaseTrack.id_count

    @staticmethod
    def clear_count():
        BaseTrack.id_count = 0


class Track(BaseTrack):
    def __init__(self, dets, predictor, interpolation):
        """
        :param detection: [6+C] -> [f,x,y,w,h,s,cid]
        """
        dets[dets[:, 5] < 0, 5] = 1
        self.dets = {int(det[0]): det[1:7] for det in dets}

        self.cid = int(dets[0, 6])
        self.f_min = int(np.min(dets[:, 0]))
        self.f_max = int(np.max(dets[:, 0]))
        self.score = np.mean(dets[:, 5])
        self.score_min = np.min(dets[:, 5])
        self.score_max = np.max(dets[:, 5])

        self.predictor = predictor
        self.interpolation = interpolation
        self.motion_predictor = eval(f'{predictor["predictor"]}Predictor')(
            predictor["delta_t"],
            predictor["preserve_scale"],
        )
        self.pred_dets = dict()
        self.track_id = self.next_id()

    def predict(self, delta_t, conf_thr=None, mode='bi-direction', delta_xy=None):
        self.pred_dets = dict()  # clean the pred_dets
        assert mode in ('bi-direction', 'forward', 'backward')
        if mode != 'forward':
            self.pred_dets.update(
                self.motion_predictor.predict(
                    dets=self.dets,
                    f_start=self.f_min - delta_t,
                    f_stop=self.f_min - 1,
                    conf_thr=conf_thr,
                    delta_xy=delta_xy,
                )
            )
        if mode != 'backward':
            self.pred_dets.update(
                self.motion_predictor.predict(
                    dets=self.dets,
                    f_start=self.f_max + 1,
                    f_stop=self.f_max + delta_t,
                    conf_thr=conf_thr,
                    delta_xy=delta_xy,
                )
            )

    def update(self, track):
        if self.f_max < track.f_min:
            missing = (self.f_max, track.f_min)
            self.f_max = track.f_max
            self.dets.update(track.dets)
        elif track.f_max < self.f_min:
            missing = (track.f_max, self.f_min)
            self.f_min = track.f_min
            self.dets.update(track.dets)
        else:
            missing = (None, None)
            for f in range(track.f_min, track.f_max + 1):
                det0 = self.dets.get(f)
                det1 = track.dets.get(f)
                if (det0 is None) and (det1 is None):
                    continue
                elif (det0 is not None) and (det1 is not None):
                    self.dets[f] = (det0 + det1) / 2
                elif det0 is not None:
                    self.dets[f] = det0
                elif det1 is not None:
                    self.dets[f] = det1
            self.f_min = min(self.f_min, track.f_min)
            self.f_max = max(self.f_max, track.f_max)

        self.pred_dets = dict()
        self.score = np.mean([det[4] for det in self.dets.values()])
        self.score_min = min([det[4] for det in self.dets.values()])
        self.score_max = max([det[4] for det in self.dets.values()])

        # perform interpolation for long enough trks
        if self.interpolation:
            start, stop = missing
            len_1, len_2  = len(self.dets), len(track.dets)
            if (start is not None) and (max(len_1, len_2) >= stop - start - 1):
                self.interpolate(*missing)

    def interpolate(self, start, stop):
        bbox1 = self.dets[start]
        bbox2 = self.dets[stop]
        delta = (bbox2 - bbox1) / (stop - start)
        for f in range(start + 1, stop):
            self.dets[f] = bbox1 + delta * (f - start)

    def __repr__(self):
        return f'Track_{self.track_id}_{self.f_min}_{self.f_max}'

    def update_with_dets(self, dets):
        self.dets.update(dets)
        self.pred_dets = dict()
        self.f_min = min(self.dets)
        self.f_max = max(self.dets)
        self.score = np.mean([det[4] for det in self.dets.values()])
        self.score_min = min([det[4] for det in self.dets.values()])
        self.score_max = max([det[4] for det in self.dets.values()])

    def smooth(self, tau=10, conf=0):
        tracks = np.array([
            [f] + det.tolist() for f, det in self.dets.items() if det[4] >= conf
        ])
        len_scale = np.clip(tau * np.log(tau ** 3 / len(tracks)), tau ** -1, tau ** 2)
        gpr = GPR(RBF(len_scale, 'fixed'))
        t = tracks[:, 0].reshape(-1, 1)
        x = tracks[:, 1].reshape(-1, 1)
        y = tracks[:, 2].reshape(-1, 1)
        w = tracks[:, 3].reshape(-1, 1)
        h = tracks[:, 4].reshape(-1, 1)
        s = tracks[:, 5].reshape(-1, 1)
        gpr.fit(t, x)
        xx = gpr.predict(t)[:, 0]
        gpr.fit(t, y)
        yy = gpr.predict(t)[:, 0]
        gpr.fit(t, w)
        ww = gpr.predict(t)[:, 0]
        gpr.fit(t, h)
        hh = gpr.predict(t)[:, 0]
        gpr.fit(t, s)
        ss = gpr.predict(t)[:, 0]
        for i in range(len(t)):
            self.dets[t[i, 0]] = np.array([xx[i], yy[i], ww[i], hh[i], ss[i], tracks[i, 5]])
