# HIT
**Hierarchical IoU Tracking based on Interval**
[![arXiv](https://img.shields.io/badge/arXiv-2406.13271-<COLOR>.svg)](https://arxiv.org/abs/2406.13271)

## Abstract

Multi-Object Tracking (MOT) aims to detect and associate all targets of given classes across frames. 
Current dominant solutions, e.g. ByteTrack and StrongSORT++, follow the hybrid pipeline, 
in which most of the associations are done online first, 
and then the results are refined using offline tricks such as interpolation and global link. 
While this paradigm offers flexibility in application, 
the disjoint design between the two stages results in suboptimal performance. 
In this paper, we propose a Hierarchical IoU Tracking (HIT) framework, 
which achieves unified hierarchical tracking by utilizing tracklet intervals as the prior. 
To ensure the conciseness, only IoU is utilized for association, while the heavy appearance models, 
tricky auxiliary cues, and learning-based association modules are discarded. 
In addition, three inconsistency issues regarding target size, camera movement and hierarchical cues are identified, 
and corresponding solutions to guarantee the reliability of associations are proposed. 
Though its simplicity, our method achieves promising performance on four public datasets, i.e., MOT17, KITTI, DanceTrack and VisDrone, 
contributing a strong baseline for future research on tracking. 
Moreover, we experiment with seven trackers and the results indicate that HIT can be seamlessly integrated with other solutions, 
whether they are motion-based, appearance-based or learning-based. 
Codes will be released at https://github.com/dyhBUPT/HIT.

![Image](https://github.com/user-attachments/assets/7c09174e-a808-46cb-aed7-439be044cd74)

## vs. SOTA

![Image](https://github.com/user-attachments/assets/ccf71995-2592-492a-9bfd-685bdd969681)

## Preparation

Download datasets MOT17, DanceTrack, KITTI and VisDrone, and link them to the folder `dataset`.
This is used for performance evaluation for the validation set (details in `opts.py` -> `GT_DIRS`).

For the running environment, you need `numpy`, `yaml`, `lap`, `filterpy`, `sklearn`, `scipy`, `tqdm`, `yaml`.
You also need [TrackEval](https://github.com/JonathonLuiten/TrackEval) for evaluation.

No GPU is needed!!!

## Tracking

#### (1) main

- **Run baseline on MOT17 val set**
  ```shell
  python main.py --cfg MOT17_val_Base --input_dir detections/MOT17-val_YOLOX_nms.8_score.1/ --exp_name MOT17_val_Base
  ```
  Then you will get:
  ```shell
  ========== Evaluation Results of PEDESTRIAN ==========
  |  HOTA |  DetA |  AssA |  MOTA |  IDF1 |  IDP  |  IDR  |  IDSW |  LocA |   FP  |   FN  |
  | 67.22 | 67.10 | 67.88 | 77.94 | 78.01 | 83.30 | 73.36 |  425  | 86.68 |  2517 |  8945 |
  ```
- **Run HIT on MOT17 val set**
  ```shell
  python main.py --cfg MOT17_val_Full --input_dir detections/MOT17-val_YOLOX_nms.8_score.1/ --exp_name MOT17_val_Full
  ```
  Then you will get:
  ```shell
  ========== Evaluation Results of PEDESTRIAN ==========
  |  HOTA |  DetA |  AssA |  MOTA |  IDF1 |  IDP  |  IDR  |  IDSW |  LocA |   FP  |   FN  |
  | 68.03 | 67.01 | 69.59 | 78.12 | 79.47 | 84.70 | 74.85 |  210  | 86.68 |  2657 |  8925 |
  ```

- **Run HIT on MOT17 test set**
  ```shell
  python main.py --cfg MOT17_test_Full --input_dir detections/MOT17-test_YOLOX_nms.8_score.1/ --exp_name MOT17_test_Full
  ```
  Then you will get the tracking results under the folder `outputs/MOT17-test/MOT17_test_Full`.

Similarly, you can run HIT on DanceTrack (val/test), KITTI (val/test) and VisDrone (val) using config files under the folder `configs`.

#### (2) hierarchical strategy

To change the hierarchical strategy from `tracklet interval` to `temporal window` (Table 5 in paper), you just need to use `main_W.py`.
In detail:
- **Run baseline (temporal window) on MOT17 val set**
  ```shell
  python main_W.py --cfg MOT17_val_Base --input_dir detections/MOT17-val_YOLOX_nms.8_score.1/ --exp_name MOT17_val_Base-W
  ```
  Then you will get:
  ```shell
  ========== Evaluation Results of PEDESTRIAN ==========
  |  HOTA |  DetA |  AssA |  MOTA |  IDF1 |  IDP  |  IDR  |  IDSW |  LocA |   FP  |   FN  |
  | 66.10 | 66.88 | 65.89 | 77.68 | 77.25 | 82.52 | 72.62 |  485  | 86.70 |  2538 |  9003 |
  ```
- **Run baseline (temporal window) on KITTI val set**
  ```shell
  python main_W.py --cfg KITTI_val_Base --input_dir detections/KITTI-val_PermaTrack --exp_name KITTI_val_Base-W
  ```
  Then you will get:
  ```shell
  ========== Evaluation Results of CAR ==========
  |  HOTA |  DetA |  AssA |  MOTA |  IDF1 |  IDP  |  IDR  |  IDSW |  LocA |   FP  |   FN  |
  | 80.32 | 80.24 | 80.70 | 89.41 | 90.02 | 89.56 | 90.49 |  118  | 89.71 |  579  |  465  |
  ```

#### (3) integration

You can integrate HIT with other trackers, for example:

- **Integrate HIT with FairMOT**
  ```shell
  python main.py --cfg MOT17_val_Combine --input_dir trackers/FairMOT/ --exp_name HIT-FairMOT
  ```
  Then you will get:
  ```shell
  ========== Evaluation Results of PEDESTRIAN ==========
  |  HOTA |  DetA |  AssA |  MOTA |  IDF1 |  IDP  |  IDR  |  IDSW |  LocA |   FP  |   FN  |
  | 59.42 | 58.98 | 60.40 | 72.45 | 74.24 | 80.21 | 69.10 |  223  | 81.88 |  3581 | 11042 |
  ```
- **Integrate HIT with ByteTrack**
  ```shell
  python main.py --cfg MOT17_val_Combine --input_dir trackers/ByteTrack/ --exp_name HIT-ByteTrack
  ```
  Then you will get:
  ```shell
  ========== Evaluation Results of PEDESTRIAN ==========
  |  HOTA |  DetA |  AssA |  MOTA |  IDF1 |  IDP  |  IDR  |  IDSW |  LocA |   FP  |   FN  |
  | 69.44 | 68.31 | 71.17 | 80.27 | 80.95 | 85.77 | 76.63 |  118  | 86.34 |  2387 |  8129 |
  ```

## Citation

```
@article{du2024hierarchical,
  title={Hierarchical IoU Tracking based on Interval},
  author={Du, Yunhao and Zhao, Zhicheng and Su, Fei},
  journal={arXiv preprint arXiv:2406.13271},
  year={2024}
}
```
>>>>>>> 52508b5... init commit
