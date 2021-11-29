# Suppress-and-Reﬁne Framework for End-to-End 3D Object Detection

This repo is the official implementation of ["Suppress-and-Refine Framework for End-to-End 3D Object Detection"](https://arxiv.org/abs/2103.10042).

A simple, fast, efficient and end-to-end 3D object detector **without NMS**. 

## Getting Started
  
## Main results
### ScanNet V2

|Method | backbone | mAP@0.25 | mAP@0.5 | Runtime (FPS) | Ckpt |
|:---:|:---:|:---:|:---:|:---:|:---:|
|[VoteNet](https://arxiv.org/abs/1904.09664) | PointNet++ | 62.9 | 39.9 | 10.8| -|
|[H3DNet](https://arxiv.org/abs/2006.05682) | 4xPointNet++ | 67.2| 48.1 | 4.4 | -|
|[MLCVNet](https://arxiv.org/abs/2004.05679) | PointNet++ | 64.5 | 41.4 | 6.7 | -|
|[BRNet](https://arxiv.org/abs/2006.05682) | PointNet++ | 66.1 |  50.9| 8.7| - |
|[Group-Free](https://arxiv.org/abs/2104.00678) | PointNet++ | 67.3 | 48.9 | 7.1|-|
|Ours | PointNet++ | 66.2 | 53.5 | **13.5** |[model_ckpt](https://1drv.ms/u/s!AoLLF1KOJvApgSrXshEbANYLWYKF)|
### SUN RGB-D

|Method | backbone | mAP@0.25 | mAP@0.5 | Ckpt |
|:---:|:---:|:---:|:---:|:---:|
|[VoteNet](https://arxiv.org/abs/1904.09664)| PointNet++ | 59.1 | 35.8  |- |
|[H3DNet](https://arxiv.org/abs/2006.05682) | 4xPointNet++ | 60.1 | 39.0 | -|
|[MLCVNet](https://arxiv.org/abs/2004.05679)|PointNet++ |  59.8 | - |  -| 
|[BRNet](https://arxiv.org/abs/2006.05682) | PointNet++ |  61.1| 43.7|  -|
|[Group-Free](https://arxiv.org/abs/2104.00678) | PointNet++ | 63.0 | 45.2 |  - |
|Ours | PointNet++ | 60.0 | 44.7| [model_ckpt](https://1drv.ms/u/s!AoLLF1KOJvApgSrXshEbANYLWYKF) |

The FPS is tested on a V100 GPU.

## Quick start


<details>
<summary>Installation</summary>

This repository is based on mmdetection3d, please follow this [page](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/getting_started.md) for installation guidance.

</details>

<details>
<summary>Reproduce our results on SCANNET and SUNRGBD</summary>

For SCANNET.

```shell
CUDA_VISIBLE_DEVICES=0,1 PORT=29600 ./tools/dist_train.sh configs/sr/scannet_baseline.py 2
CUDA_VISIBLE_DEVICES=2,3 PORT=29601 ./tools/dist_train.sh configs/sr/scannet_baseline.py 2
CUDA_VISIBLE_DEVICES=4,5 PORT=29602 ./tools/dist_train.sh configs/sr/scannet_baseline.py 2
CUDA_VISIBLE_DEVICES=6,7 PORT=29603 ./tools/dist_train.sh configs/sr/scannet_baseline.py 2
```

For SUNRGBD
```shell
CUDA_VISIBLE_DEVICES=0,1 PORT=29600 ./tools/dist_train.sh configs/sr/sunrgbd_baseline.py 2
```

</details>


<details>
<summary>Evaluation</summary>

Please first download the ckpt from the ckpt link provided above.

Then for SCANNET.

```shell
./tools/dist_test.sh configs/sr/scannet_baseline.py epoch_30.pth 2 --eval mAP
```

For SUNRGBD
```shell
./tools/dist_test.sh configs/sr/sunrgbd_baseline.py epoch_33.pth 4 --eval mAP
```

</details>

## Acknowledgement

Our code is based on wonderful [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). Very apperciate their works!

## Citation

If you find this project useful in your research, please consider cite:

```
@article{liu2021suppress,
  title={Suppress-and-Refine Framework for End-to-End 3D Object Detection},
  author={Liu, Zili and Xu, Guodong and Yang, Honghui and Chen, Minghao and Wu, Kuoliang and Yang, Zheng and Liu, Haifeng and Cai, Deng},
  journal={arXiv preprint arXiv:2103.10042},
  year={2021}
}
```