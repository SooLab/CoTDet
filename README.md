CoTDet: Affordance Knowledge Prompting for Task Driven Object Detection
========

[Jiajin Tang*](https://toneyaya.github.io/), Ge Zheng*, [Jingyi Yu](https://vic.shanghaitech.edu.cn/vrvc/en/people/u), and [Sibei Yang](https://faculty.sist.shanghaitech.edu.cn/yangsibei/). *denotes equal contribution. This repository is the official implementation of our [CoTDet](https://arxiv.org/abs/2309.01093).

***

## Overview
In this paper, we focus on challenging task driven object detection, which is practical in the real world yet underexplored. To bridge the gap between abstract task requirements and objects in the image, we propose to explicitly extract visual affordance knowledge for the task and detect objects having consistent visual attributes to the visual knowledge. Furthermore, our CoTDet utilizes visual affordance knowledge to condition the decoder in localizing and recognizing suitable objects.


![avatar](details/framework.png)


***


## Getting Strated
**1. Prepare the code and the environment**
Git clone our repository, creating a python environment and activate it via the following command
``` bash
# clone repository
git clone https://github.com/Toneyaya/CoTDet.git
# create conda environment
cd CoTDet
conda env create -f environment.yaml
# install detectron2 
python -m pip install -e detectron2
# install MultiScaleDeformableAttention
cd cotdet/modeling/pixel_encoder/ops
sh make.sh
```
**2. Download the [images](https://github.com/coco-tasks/dataset?tab=readme-ov-file#image-lists) and place them (both train and test) in the directory:**
```
CoTDet
├── datasets
  ├── coco-tasts
    ├── annotations
    ├── images
      ├── 1.jpg
      ├── 2.jpg
      ├── 3.jpg
      ├── ...
```
**3. Download our pre-trained [weights](https://drive.google.com/file/d/1gbLC1obJg9rYFwdTquwLiRJb22nj9nuQ/view?usp=sharing) which is pre-trained on a subset of the coco data removing all images that are duplicates of coco-tasks. Then put the pretrain weight path [here](configs/COCOTASK_R101.yaml#3) at line 3.**
***
## Training

```python
OPENBLAS_NUM_THREADS=1 python train_net.py --num-gpus 8 --config-file configs/COCOTASK_R101.yaml
```

### Evaluation
You can download our model [here](https://drive.google.com/file/d/16mlb35W94smyPYMcAv2LhEaLXRCEsWJn/view?usp=sharing) and enter the paths for evaluation. Of course, you can also evaluate your training results in the same way.
```python
OPENBLAS_NUM_THREADS=1 python train_net.py --num-gpus 8 --config-file configs/COCOTASK_R101.yaml --eval-only MODEL.WEIGHT ckpt_path
```
***
## Results


If you find our work helpful for your research, please consider citing the following BibTeX entry.

```BibTeX
@InProceedings{Tang_2023_ICCV,
    author    = {Tang, Jiajin and Zheng, Ge and Yu, Jingyi and Yang, Sibei},
    title     = {CoTDet: Affordance Knowledge Prompting for Task Driven Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {3068-3078}
}
```

## Acknowledgement

Many thanks to these excellent opensource projects 
* [detectron2](https://github.com/facebookresearch/detectron2).
* [DN-DETR](https://github.com/IDEA-Research/DN-DETR)
* [MaskDINO](https://github.com/IDEA-Research/MaskDINO)

