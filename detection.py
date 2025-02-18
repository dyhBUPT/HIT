import numpy as np
from copy import copy
from os.path import join


class Detection:
    def __init__(self, input_path, conf_thr, nms_thr=None):
        """
        detections: [f,-1,x,y,w,h,s,-1,-1,cid]
        """
        detections = self.load_detections(input_path)
        detections = np.delete(detections, [1, 7, 8], axis=1)  # [fid,x,y,w,h,s,cid]
        detections = detections[detections[:, 5] >= conf_thr]
        if nms_thr:
            detections = self.nms(detections, nms_thr, 'nms')
        self.detections = detections[
            np.lexsort([
                detections[:, 1],
                detections[:, 0],
            ])
        ]  # sort by `fid` & `x`

    def load_detections(self, path):
        if path.endswith('.txt'):
            detections = np.loadtxt(path, delimiter=',')
        elif path.endswith('.npy'):
            detections = np.load(path)

        return detections

    @property
    def frames(self):
        min_frame = int(min(self.detections[:, 0]))
        max_frame = int(max(self.detections[:, 0]))
        return min_frame, max_frame

    def nms(self, detections, iou_thr=0.5, mode='nms'):
        """
        Note: this NMS treats all classes equally.
        """

        def frame_level_nms(bboxes, iou_thr, mode):
            """
            :param bboxes: [N,5] -> [x,y,w,h,s]
            :param iou_thr:
            :param mode:
            :return:
            """
            bboxes = copy(bboxes)
            bboxes[:, 2:4] += bboxes[:, :2]  # [x1,y1,x2,y2,s]
            x1, y1, x2, y2, scores = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3], bboxes[:, 4]
            areas = (x2 - x1) * (y2 - y1)
            num_bboxes = bboxes.shape[0]
            keep = list()  # 存储保留下的框
            if mode == 'nms':
                index_sorted = np.argsort(scores)[::-1]  # nms算法中需先对bbox按得分排序
                while index_sorted.shape[0]:  # 按得分顺序遍历所有框
                    '''计算IoU'''
                    index_max = index_sorted[0]  # 最高分框
                    x1_ = np.maximum(x1[index_max], x1[index_sorted])
                    y1_ = np.maximum(y1[index_max], y1[index_sorted])
                    x2_ = np.minimum(x2[index_max], x2[index_sorted])
                    y2_ = np.minimum(y2[index_max], y2[index_sorted])
                    w_ = np.maximum(0., x2_ - x1_)
                    h_ = np.maximum(0., y2_ - y1_)
                    inter = w_ * h_
                    iou = inter / (areas[index_max] + areas[index_sorted] - inter)
                    '''框过滤 & 列表更新'''
                    index_sorted = index_sorted[
                        np.where(iou <= iou_thr)[0]
                    ]
                    keep.append(index_max)
            elif mode == 'soft_nms':
                index_left = np.arange(num_bboxes)  # 所有当前bbox的索引
                keep = [i for i in range(num_bboxes)]  # 直接返回所有框的索引
                for _ in range(num_bboxes):
                    '''计算IoU'''
                    index_max = np.argmax(scores[index_left])  # 最高分框
                    index_max = index_left[index_max]  # index_max是相对index_left的索引
                    index_left = index_left[index_left != index_max]  # 其余框
                    x1_ = np.maximum(x1[index_max], x1[index_left])
                    y1_ = np.maximum(y1[index_max], y1[index_left])
                    x2_ = np.minimum(x2[index_max], x2[index_left])
                    y2_ = np.minimum(y2[index_max], y2[index_left])
                    w_ = np.maximum(0., x2_ - x1_)
                    h_ = np.maximum(0., y2_ - y1_)
                    inter = w_ * h_
                    iou = inter / (areas[index_max] + areas[index_left] - inter)
                    '''重打分'''
                    index_rescore = iou > iou_thr  # 需要重打分的框索引
                    weights = 1 - iou[index_rescore]  # 重打分权重
                    scores[index_left[index_rescore]] = weights * scores[index_left[index_rescore]]
            else:
                raise RuntimeError('mode error')
            return np.array(keep)

        frames = set(detections[:, 0])
        detections_out = np.empty([0, detections.shape[1]])
        for frame in frames:
            detections_frame = detections[detections[:, 0] == frame]
            keep = frame_level_nms(detections_frame[:, 1:], iou_thr, mode)
            detections_out = np.append(detections_out, detections_frame[keep], axis=0)
        return detections_out
