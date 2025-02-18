"""
Hierarchical IoU Tracking based on tracklet interval
"""
from tqdm import tqdm
from os.path import join

from opts import opt
from tracker import Tracker
from detection import Detection
from evaluation import evaluation


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

        tracker = Tracker(
            dataset=args.dataset,
            video=video,
            detections=detections,
            high_conf_thr=args.params['conf_thr']['high'],
            motion_param=args.params['motion_matching'],
            iou_param=args.params['iou_matching'],
            predictor=args.params['predictor'],
            interpolation=args.params['interpolation'],
        )

        for i, delta_t in enumerate(args.params['delta_t'], start=1):
            if 1 < i == len(args.params['delta_t']):
                tracker.allow_overlapping = True

            tracker.before_match(delta_t)
            if delta_t == 1:
                tracker.camera_motion_compensation(**args.params['ConsistentCamera'])
                tracker.pre_match(args.params['ConsistentMotion'])

            for frame in range(tracker.frames[0], tracker.frames[1]):
                tracker.match(frame)
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
