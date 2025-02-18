import os
import sys
import shutil
import numpy as np
from copy import copy
from os.path import split, join

import trackeval
from opts import opt


def VisDrone_clean_gt_and_pred(gt_dir, pred_dir, tracker):
    """ref: https://github.com/alibaba/u2mot/blob/main/tools/utils/eval_visdrone.py"""
    valid_cls = [1, 4, 5, 6, 9]
    ignore_cls = [0, 11]
    tmp_gt_dir = f'{gt_dir}_tmp'
    tmp_tracker = f'{tracker}_tmp'
    os.makedirs(tmp_gt_dir, exist_ok=True)
    os.makedirs(join(pred_dir, tmp_tracker), exist_ok=True)

    for file in os.listdir(gt_dir):
        gt = np.loadtxt(join(gt_dir, file), delimiter=',')
        pred = np.loadtxt(join(pred_dir, tracker, file), delimiter=',')

        # ignored regions filtering
        new_gt, new_pred = [], []
        for frame in set(gt[:, 0]):
            # load frame-level gt/pred
            gt_frame = gt[gt[:, 0] == frame]
            gt_cx_cy = gt_frame[:, 2:4] + gt_frame[:, 4:6] / 2
            pred_frame = pred[pred[:, 0] == frame]
            pred_cx_cy = pred_frame[:, 2:4] + pred_frame[:, 4:6] / 2
            # get ignored regions
            ignores = gt_frame[np.isin(gt_frame[:, 7], ignore_cls)]
            # initialize ignoring flags
            gt_flag = np.zeros(len(gt_frame), dtype=bool)
            pred_flag = np.zeros(len(pred_frame), dtype=bool)
            # use every ignore region to modify ignoring flags
            for ignore_det in ignores:
                ig_x1, ig_y1, ig_w, ig_h = ignore_det[2:6]
                ig_x2, ig_y2 = ig_x1 + ig_w, ig_y1 + ig_h
                gt_flag += (ig_x1 < gt_cx_cy[:, 0]) * (gt_cx_cy[:, 0] < ig_x2) * \
                          (ig_y1 < gt_cx_cy[:, 1]) * (gt_cx_cy[:, 1] < ig_y2)
                pred_flag += (ig_x1 < pred_cx_cy[:, 0]) * (pred_cx_cy[:, 0] < ig_x2) * \
                          (ig_y1 < pred_cx_cy[:, 1]) * (pred_cx_cy[:, 1] < ig_y2)
            # filtering
            new_gt.extend(gt_frame[~gt_flag])
            new_pred.extend(pred_frame[~pred_flag])
        gt = np.array(new_gt)
        pred = np.array(new_pred)

        # class & confidence filtering
        gt = gt[np.isin(gt[:, 7], valid_cls)]
        gt = gt[gt[:, 6] > 0]
        gt[:, 7] = 1
        pred = pred[np.isin(pred[:, 9], valid_cls)]
        pred = pred[pred[:, 6] > 0]
        pred[:, 9] = 1

        # save new gts / preds
        np.savetxt(
            join(tmp_gt_dir, file),
            gt,
            fmt='%d,%d,%d,%d,%d,%d,%d,%d,%d,%d',
        )
        np.savetxt(
            join(pred_dir, tmp_tracker, file),
            pred,
            fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d',
        )

    return tmp_gt_dir, pred_dir, tmp_tracker


class HiddenPrints:
    def __init__(self, enable):
        self.enable = enable

    def __enter__(self):
        if self.enable:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            sys.stdout.close()
            sys.stdout = self._original_stdout


def get_dataset_cfg(gt_folder, tracker_folder, tracker_name, seqmap, gt_file, classes, dataset='MOT17', seq_info=None):
    """Get default configs for trackeval.datasets.MotChallenge2DBox."""
    dataset_config = dict(
        # Location of GT data
        GT_FOLDER=gt_folder,
        # Trackers location
        TRACKERS_FOLDER=tracker_folder,
        # Where to save eval results
        # (if None, same as TRACKERS_FOLDER)
        OUTPUT_FOLDER=None,
        # the default tracker
        TRACKERS_TO_EVAL=[tracker_name],
        # Option values: ['pedestrian']
        CLASSES_TO_EVAL=classes,
        # Option Values: 'MOT15', 'MOT16', 'MOT17', 'MOT20'
        BENCHMARK=dataset,
        # Option Values: 'train', 'test'
        SPLIT_TO_EVAL='train',
        # Whether tracker input files are zipped
        INPUT_AS_ZIP=False,
        # Whether to print current config
        PRINT_CONFIG=True,
        # Whether to perform preprocessing
        # (never done for MOT15)
        DO_PREPROC=False if dataset == 'MOT15' else True,
        # Tracker files are in
        # TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        TRACKER_SUB_FOLDER='',
        # Output files are saved in
        # OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        OUTPUT_SUB_FOLDER='',
        # Names of trackers to display
        # (if None: TRACKERS_TO_EVAL)
        TRACKER_DISPLAY_NAMES=None,
        # Where seqmaps are found
        # (if None: GT_FOLDER/seqmaps)
        SEQMAP_FOLDER=None,
        # Directly specify seqmap file
        # (if none use seqmap_folder/benchmark-split_to_eval)
        # SEQMAP_FILE=f'{gt_folder}/../trackeval/seqmap.txt',
        SEQMAP_FILE=seqmap,
        # If not None, specify sequences to eval
        # and their number of timesteps
        SEQ_INFO=seq_info,
        # '{gt_folder}/{seq}.txt'
        GT_LOC_FORMAT=gt_file,
        # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
        # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
        # If True, the middle 'benchmark-split' folder is skipped for both.
        SKIP_SPLIT_FOL=True,
    )
    return dataset_config


def get_eval_results(results):
    eval_results = dict()
    if 'HOTA' in results:
        eval_results['HOTA'] = np.average(results['HOTA']['HOTA'])
        eval_results['AssA'] = np.average(results['HOTA']['AssA'])
        eval_results['DetA'] = np.average(results['HOTA']['DetA'])
        eval_results['AssRe'] = np.average(results['HOTA']['AssRe'])
        eval_results['AssPr'] = np.average(results['HOTA']['AssPr'])
        eval_results['DetRe'] = np.average(results['HOTA']['DetRe'])
        eval_results['DetPr'] = np.average(results['HOTA']['DetPr'])
        eval_results['LocA'] = np.average(results['HOTA']['LocA'])

    if 'CLEAR' in results:
        eval_results['MOTA'] = np.average(results['CLEAR']['MOTA'])
        eval_results['MOTP'] = np.average(results['CLEAR']['MOTP'])
        eval_results['IDSW'] = np.average(results['CLEAR']['IDSW'])
        eval_results['TP'] = np.average(results['CLEAR']['CLR_TP'])
        eval_results['FP'] = np.average(results['CLEAR']['CLR_FP'])
        eval_results['FN'] = np.average(results['CLEAR']['CLR_FN'])
        eval_results['Frag'] = np.average(results['CLEAR']['Frag'])
        eval_results['MT'] = np.average(results['CLEAR']['MT'])
        eval_results['ML'] = np.average(results['CLEAR']['ML'])

    if 'Identity' in results:
        eval_results['IDF1'] = np.average(results['Identity']['IDF1'])
        eval_results['IDTP'] = np.average(results['Identity']['IDTP'])
        eval_results['IDFN'] = np.average(results['Identity']['IDFN'])
        eval_results['IDFP'] = np.average(results['Identity']['IDFP'])
        eval_results['IDP'] = np.average(results['Identity']['IDP'])
        eval_results['IDR'] = np.average(results['Identity']['IDR'])

    return eval_results


def print_eval_results(results):
    metrics = ['HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1', 'IDP', 'IDR', 'LocA', 'AssRe', 'AssPr', 'DetRe', 'DetPr']
    RESULTS = dict()
    for metric in metrics:
        if metric in results:
            RESULTS[metric] = results[metric] * 100
    metrics = ['FP', 'FN', 'IDSW', 'IDTP', 'IDFP', 'IDFN']
    for metric in metrics:
        if metric in results:
            RESULTS[metric] = int(results[metric])
    # print the formatted results
    metrics = ['HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1', 'IDP', 'IDR', 'IDSW', 'LocA', 'FP', 'FN']
    title, results = '|', '|'
    for metric in metrics:
        title += metric.center(7) + '|'
        result = RESULTS[metric]
        if isinstance(result, float):
            results += ('%.2f' % result).center(7) + '|'
        elif isinstance(result, int):
            results += ('%d' % result).center(7) + '|'
    print(title)
    print(results)


def evaluation(pred_dir=None):
    gt_dir = opt.gt_dir
    seqmap = opt.seqmap
    pred_dir = opt.output_dir if pred_dir is None else pred_dir
    pred_dir, tracker = split(pred_dir)

    if opt.dataset == 'MOT17':
        gt_file = '{gt_folder}/{seq}/gt/gt_half-val.txt'
        data_name = 'MotChallenge2DBox'
        classes = ['pedestrian']
        seq_info = None
    elif opt.dataset == 'DanceTrack':
        gt_file = '{gt_folder}/{seq}/gt/gt.txt'
        data_name = 'MotChallenge2DBox'
        classes = ['pedestrian']
        seq_info = None
    elif opt.dataset == 'KITTI':
        gt_file = 'xxx.txt'
        data_name = 'Kitti2DBox'
        classes = ['car']
        seq_info = None
    elif opt.dataset == 'VisDrone':
        gt_file = '{gt_folder}/{seq}.txt'
        data_name = 'MotChallenge2DBox'
        classes = ['pedestrian']
        img_dir = f'{gt_dir}/sequences'
        gt_dir = f'{gt_dir}/annotations'
        seq_info = {
            seq: len(os.listdir(join(img_dir, seq)))
            for seq in os.listdir(img_dir)
        }
        gt_dir, pred_dir, tracker = VisDrone_clean_gt_and_pred(gt_dir, pred_dir, tracker)

    with HiddenPrints(enable=True):
        eval_config = trackeval.Evaluator.get_default_eval_config()
        dataset_config = get_dataset_cfg(gt_dir, pred_dir, tracker, seqmap, gt_file, classes, seq_info=seq_info)
        evaluator = trackeval.Evaluator(eval_config)

        dataset = [eval(f'trackeval.datasets.{data_name}')(dataset_config)]
        metrics = [
            getattr(trackeval.metrics, metric)(dict(METRICS=[metric], THRESHOLD=0.5))
            for metric in ['HOTA', 'CLEAR', 'Identity']
        ]
        output_res, _ = evaluator.evaluate(dataset, metrics)

    for cls in classes:
        output = output_res[data_name][tracker]['COMBINED_SEQ'][cls]
        eval_results = get_eval_results(output)

        if opt.dataset == 'VisDrone':
            cls = 'all 5 classes'
            shutil.rmtree(gt_dir)
            shutil.rmtree(join(pred_dir, tracker))

        print(f'========== Evaluation Results of {cls.upper()} ==========')
        print_eval_results(eval_results)
