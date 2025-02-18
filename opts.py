import os
import json
import yaml
import argparse
from os.path import join


DATA = {
    'MOT17': {
        'val':[
            'MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN',
            'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN',
        ],
        'test':[
            'MOT17-01-FRCNN', 'MOT17-03-FRCNN', 'MOT17-06-FRCNN', 'MOT17-07-FRCNN',
            'MOT17-08-FRCNN', 'MOT17-12-FRCNN', 'MOT17-14-FRCNN',
        ],
    },
    'MOT20': {
        'test':[
            'MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08',
        ],
    },
    'DanceTrack': {
        'val': [
            'dancetrack0004', 'dancetrack0005', 'dancetrack0007', 'dancetrack0010', 'dancetrack0014',
            'dancetrack0018', 'dancetrack0019', 'dancetrack0025', 'dancetrack0026', 'dancetrack0030',
            'dancetrack0034', 'dancetrack0035', 'dancetrack0041', 'dancetrack0043', 'dancetrack0047',
            'dancetrack0058', 'dancetrack0063', 'dancetrack0065', 'dancetrack0073', 'dancetrack0077',
            'dancetrack0079', 'dancetrack0081', 'dancetrack0090', 'dancetrack0094', 'dancetrack0097'
        ],
        'test': [
            'dancetrack0003', 'dancetrack0009', 'dancetrack0011', 'dancetrack0013', 'dancetrack0017',
            'dancetrack0021', 'dancetrack0022', 'dancetrack0028', 'dancetrack0031', 'dancetrack0036',
            'dancetrack0038', 'dancetrack0040', 'dancetrack0042', 'dancetrack0046', 'dancetrack0048',
            'dancetrack0050', 'dancetrack0054', 'dancetrack0056', 'dancetrack0059', 'dancetrack0060',
            'dancetrack0064', 'dancetrack0067', 'dancetrack0070', 'dancetrack0071', 'dancetrack0076',
            'dancetrack0078', 'dancetrack0084', 'dancetrack0085', 'dancetrack0088', 'dancetrack0089',
            'dancetrack0091', 'dancetrack0092', 'dancetrack0093', 'dancetrack0095', 'dancetrack0100'
        ],
    },
    'KITTI': {
        'val': ['%04d' % i for i in range(0, 21)],  # 0000 ~ 0020
        'test': ['%04d' % i for i in range(0, 29)],  # 0000 ~ 0028
    },
    'VisDrone': {
        'val': [
            'uav0000009_03358_v', 'uav0000073_00600_v', 'uav0000073_04464_v', 'uav0000077_00720_v',
            'uav0000088_00290_v', 'uav0000119_02301_v', 'uav0000120_04775_v', 'uav0000161_00000_v',
            'uav0000188_00000_v', 'uav0000201_00000_v', 'uav0000249_00001_v', 'uav0000249_02688_v',
            'uav0000297_00000_v', 'uav0000297_02761_v', 'uav0000306_00230_v', 'uav0000355_00001_v',
            'uav0000370_00001_v',
        ],
    }
}

GT_DIRS = {
    'MOT17': {
        'val': 'dataset/MOT17/train',
    },
    'DanceTrack': {
        'val': 'dataset/DanceTrack/val',
    },
    'KITTI': {
        'val': 'dataset/KITTI/kitti_tracking/kitti_2d_box_val_half',
    },
    'VisDrone': {
        'val': 'dataset/VisDroneMOT/VisDrone2019-MOT-test-dev',
    }
}


# The default settings for MOT17
PARAMS = {
    # If true, load tracks instead of detections as inputs.
    'input_tracks': False,
    # The timestep for hierarchical tracking.
    'delta_t': [1, 5, 10, 15, 20, 30, 5],
    # The confidence threshold for tracking.
    'conf_thr': {
        # The minimum score while loading detections.
        # If you want to use `BYTE`, please set it to a lower value than `high`.
        # Otherwise, please set it to the same value as `high`.
        'low': .1,
        # The score threshold for the first-stage tracking.
        'high': .6,
    },
    # If specified (float 0~1), perform NMS while loading detections.
    'nms_thr': None,
    # If set `enable` to `True`, estimate camera motion for more accurate matching.
    'ConsistentCamera': {
        'enable': True,
        'thr': .65,  # the IoU threshold to recognize camera movement.
        'tau': 100,  # the temperature factor for gaussian smoothing.
    },
    # If set to `True`, apply pre-matching to equip detections with motion information.
    'ConsistentMotion': True,
    # The motion predictor for `delta_t > 1`, i.e., Linear or Kalman
    'predictor': {
        'predictor': 'Kalman',  # 'Kalman' or 'Linear'
        'delta_t': 5,  # the time span for prediction
        # For `Linear`, preserve `w,h`;
        # For `Kalman`, preserve `scale` for missing observation.
        'preserve_scale': False,
    },
    # The basic settings for motion-based matching
    'motion_matching': {
        'mode': 'cons_hm_iou',  # `iou` / `hm_iou` / `cons_hm_iou`
        'thr': .2,  # the minimum iou for association
        'cons_w': 64,  # the maximum width for cons_iou
        'cons_tau': .2,  # the temperature factor for cons_iou
        'cons_score': .6,  # the minimum score for cons_iou (set it to the same as `conf_thr['high']`)
    },
    # The basic settings for iou-based matching
    'iou_matching': {
        'mode': 'cons_hm_iou',  # `iou` / `hm_iou` / `cons_hm_iou`
        'thr': .2,  # the minimum iou for association
        'cons_w': 64,  # the maximum width for cons_iou
        'cons_tau': .2,  # the temperature factor for cons_iou
        'cons_score': .6,  # the minimum score for cons_iou (set it to the same as `conf_thr['high']`)
    },
    # If set to `True`, apply linear interpolation while updating tracks.
    'interpolation': False,
    # The tracklet-level post-processing.
    'post_trk': {
        'min_len': None,  # the minimum length of the final tracks.
        'merge': None,
        'smooth': False,
    },
    # The detection-level post-processing as in ByteTrack.
    'post_det': {
        'min_area': -1,  # the minimum area of bboxes.
        'max_ratio': 1e5,  # the maximum w/h of bboxes.
    },
}


class opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            '--cfg',
            type=str,
            help='the config file',
            default='tmp',
        )
        self.parser.add_argument(
            '--input_dir',
            type=str,
            help='the directory of input detections',
            default='tmp',
        )
        self.parser.add_argument(
            '--exp_name',
            type=str,
            help='the experiment name',
            default='tmp',
        )

    def read_cfg(self, cfg):
        cfg = open(f'configs/{cfg}.yaml', 'r').read()
        cfg = yaml.load(cfg, Loader=yaml.FullLoader)
        return cfg

    def parse(self):
        opt = self.parser.parse_args()

        cfg = self.read_cfg(opt.cfg)
        opt.dataset = cfg['DATASET']
        opt.split = cfg['SPLIT']

        opt.videos = DATA[opt.dataset][opt.split]
        opt.params = PARAMS
        opt.params.update(cfg['PARAMS'])

        if opt.split == 'val':
            opt.gt_dir = GT_DIRS[opt.dataset][opt.split]
            opt.seqmap = join('dataset', f'{opt.dataset}-{opt.split}.txt')

        if opt.input_dir is None:
            opt.input_dir = join('inputs', f'{opt.dataset}-{opt.split}')

        if opt.exp_name is None:
            opt.exp_name = opt.cfg.split('_')[-1]

        opt.output_dir = join('outputs', opt.dataset + '-' + opt.split, opt.exp_name)
        os.makedirs(opt.output_dir, exist_ok=True)

        return opt

opt = opts().parse()
