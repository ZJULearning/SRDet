# Suppress-and-Reﬁne Framework for End-to-End 3D Object Detection

A simple, fast, efficient and end-to-end 3D object detector **without NMS**. 
  
## Main results
### ScanNet V2

|Method | backbone | mAP@0.25 | mAP@0.5 | Runtime (FPS) | Ckpt |
|:---:|:---:|:---:|:---:|:---:|:---:|
|[VoteNet](https://arxiv.org/abs/1904.09664) | PointNet++ | 62.9 | 39.9 | 10.8| -|
|[H3DNet](https://arxiv.org/abs/2006.05682) | 4xPointNet++ | 67.2| 48.1 | 4.4 | -|
|[MLCVNet](https://arxiv.org/abs/2004.05679) | PointNet++ | 64.5 | 41.4 | 6.7 | -|
|[BRNet](https://arxiv.org/abs/2006.05682) | PointNet++ | 66.1 |  50.9| 8.7| - |
|[Group-Free](https://arxiv.org/abs/2104.00678) | PointNet++ | 67.3 | 48.9 | 7.1|-|
|Ours | PointNet++ | 66.2 | 53.5 | **13.5** |[model_ckpt](https://1drv.ms/u/s!AoLLF1KOJvApgSkw6GsketVWnND9?e=mnea2u)|
### SUN RGB-D

|Method | backbone | mAP@0.25 | mAP@0.5 | Ckpt |
|:---:|:---:|:---:|:---:|:---:|
|[VoteNet](https://arxiv.org/abs/1904.09664)| PointNet++ | 59.1 | 35.8  |- |
|[H3DNet](https://arxiv.org/abs/2006.05682) | 4xPointNet++ | 60.1 | 39.0 | -|
|[MLCVNet](https://arxiv.org/abs/2004.05679)|PointNet++ |  59.8 | - |  -| 
|[BRNet](https://arxiv.org/abs/2006.05682) | PointNet++ |  61.1| 43.7|  -|
|[Group-Free](https://arxiv.org/abs/2104.00678) | PointNet++ | 63.0 | 45.2 |  - |
|Ours | PointNet++ | 60.0 | 44.7| [model_ckpt](https://1drv.ms/u/s!AoLLF1KOJvApgSpGbDbECuKCc_DG) |

The FPS is tested on a V100 GPU.

## Quick start

Our code will be released soon.

## Acknowledgement

Our code is based on wonderful [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). Very apperciate their works!

## Citation

If you find this project useful in your research, please consider cite.
