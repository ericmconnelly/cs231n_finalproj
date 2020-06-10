# A Pytorch Implementation of FSOD

### Requirements

Tested under python3.

- python packages
  - pytorch==0.4.1
  - torchvision>=0.2.0
  - cython
  - matplotlib
  - numpy
  - scipy
  - opencv
  - pyyaml
  - packaging
  - [pycocotools](https://github.com/cocodataset/cocoapi)  — for COCO dataset, also available from pip.
  - tensorboardX  — for logging the losses in Tensorboard
- An NVIDAI GPU and CUDA 9.0 are required. (Do not use other versions)
- **NOTICE**: different versions of Pytorch package have different memory usages.

### Compilation

Compile the CUDA code:

```
cd lib  # please change to this directory
sh make.sh
```


### Data Preparation

Please add `data` in the `fsod` directory and the structure is :

## Download FSOD:

  Download the images and annotations from [Google Driver](https://drive.google.com/drive/folders/1XXADD7GvW8M_xzgFpHfudYDYtKtDgZGM?usp=sharing)

## FSOD Dataset Format and Usage:

  The FSOD dataset is in MS COCO format (under debug), so place the FSOD dataset as the COCO dataset. And you can use the FSOD dataset like COCO dataset.
  
  Put the FSOD dataset as the following structure:
  ```
  YOUR_PATH
      └── your code dir
            ├── your code
            ├── ...
            │ 
            └── datasets
                  ├──── fsod
                  |       ├── annotations
                  │       │       ├── fsod_train.json
                  │       │       └── fsod_test.json
                  │       └── images
                  │             ├── part_1
                  │             └── part_2
                  │ 
                  ├──── coco
                  |       ├── annotations
                  │       │       ├── instances_train2017.json
                  │       │       └── instances_val2017.json
                  │       └── images
                  │ 
                  └── other datasets
  ```  
## Dataset Summary:


|  | Train | Test |
| ---------- | :-----------:  | :-----------: |
|No. Class | 800 | 200 |
|No. Image | 52350 | 14152 |
|No. Box | 147489 | 35102 |
|Avg No. Box / Img  | 2.82 | 2.48 |
|Min No. Img / Cls  | 22 | 30 |
|Max No. Img / Cls  | 208 | 199 |
|Avg No. Img / Cls  | 75.65 | 74.31 |
|Box Size | [6, 6828] | [13, 4605] |
|Box Area Ratio | [0.0009, 1] | [0.0009, 1] |
|Box W/H Ratio | [0.0216, 89] | [0.0199, 51.5] |

### Training and evaluation

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net_step.py --save_dir fsod_save_dir --dataset fsod --cfg configs/fsod/voc_e2e_faster_rcnn_R-50-C4_1x_old_1.yaml --bs 4 --iter_size 2 --nw 4 --load_detectron data/pretrained_model/model_final.pkl

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/test_net.py --multi-gpu-testing --dataset fsod --cfg configs/fsod/voc_e2e_faster_rcnn_R-50-C4_1x_old_1.yaml --load_ckpt Outputs/fsod_save_dir/ckpt/model_step59999.pth
```


This repository is originally built on [roytseng-tw/Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch).

## Citation

  If you use this dataset in your research, please cite this [paper](https://arxiv.org/pdf/1908.01998v1.pdf).

  ```
  @inproceedings{fan2020fsod,
    title={Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector},
    author={Fan, Qi and Zhuo, Wei and Tang, Chi-Keung and Tai, Yu-Wing},
    booktitle={CVPR},
    year={2020}
  }
  ```



