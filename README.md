
# [OptiSAR-Net: A Cross-Domain Ship Detection Method for Multi-Source Remote Sensing Data](https://ieeexplore.ieee.org/document/10757443)

An end-to-end cross-domain multi-source ship detection network inspired by the mechanisms of the human visual cortex.


<p align="center">
<img src="images/fig1.png" width="70%">
</p>
<b><p align="center" style="margin-top: -20px;">
General framework diagram of OptiSAR-Net </b></p>

[OptiSAR-Net: A Cross-Domain Ship Detection Method for Multi-Source Remote Sensing Data](https://ieeexplore.ieee.org/document/10757443).\
Jun Dong, Jiewen Feng, Xiaoyu Tang
[![MIT License][license-shield]][license-url]

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Optical and synthetic aperture radar (SAR) remote sensing are crucial for ship detection. Integrating SAR’s all-weather imaging with optical data’s shape recognition enhances downstream applications. However, current cross-domain methods often use unsupervised or semi-supervised techniques for single-source detection, limiting their practical use in cross-domain ship detection. Inspired by human visual cortex mechanisms, this paper proposes OptiSAR-Net, an end-to-end cross-domain multi-source ship detection network. Specifically, OptiSAR-Net features dual adaptive attention (DAA) for extracting standard features from SAR and optical images, and bilevel routing deformable spatial pyramid pooling-fast (BSPPF) for adapting to multiscale changes. To mitigate SAR noise, we employ VoV-GSCSP with spatial shuffling attention (VSSA) in the neck. OptiSAR-Net achieved state-of-the-art average precisions (APs) of 88.6% and 91.3% on the optical datasets DOTA and HRSC2016, respectively, and showed strong performance on the SAR datasets HRSID and SSDD. On the cross-domain heterogeneous dataset (CDHD), OptiSAR-Net differentiated ship targets effectively with only 2.7 million parameters and 11.7 GFLOPs, achieving an inference speed of 89 FPS on an NVIDIA RTX 3090. These results demonstrate that cross-domain multi-source detection significantly enhances performance and application potential compared to single-source detection.
</details>

## 📖How to use
### 📚Dataset
Download the optical dataset HRSC2016 from [HRSC2016](https://sites.google.com/site/hrsc2016/) (The training set, validation set, and test set contain 436, 181, and 444 images, respectively.).

Download the SAR dataset HRSID from [HRSID](https://drive.google.com/open?id=1BZTU8Gyg20wqHXtBPFzRazn_lEdvhsbE) (including a total of 5604 high-resolution SAR images).

**A cross-domain heterogeneous dataset (CDHD)** was constructed for training and evaluating the cross-domain multi-source detection performance of the model. This dataset retains all original images and labels from HRSC2016 and includes 1061 images randomly extracted from HRSID. **The ratio of optical images to SAR images is 1:1.**

The structure of the folders in the entire dataset is as follows:
```
DATAROOT
└── CDHD
    ├── images
    │   ├── trains (Optical: SAR = 1:1, total 832 images)
    │   │   ├──HRSC2016_001.jpg
    │   │   ├──HRSID_001.jpg
    │   │   ├──...
    │   ├── vals (Optical: SAR = 1:1, total 362 images)
    │   │   ├──HRSC2016_000.jpg
    │   │   ├──HRSID_000.jpg
    │   │   ├──...
    └──labels
    │   ├── trains
    │   │   ├──HRSC2016_001.txt
    │   │   ├──HRSID_001.txt
    │   │   ├──...
    │   ├── vals
    │   │   ├──HRSC2016_000.txt
    │   │   ├──HRSID_000.txt
    │   │   ├──...
   ...
```
Enter the address and label name of your data set in the [CDHD.yaml](./ultralytics/cfg/datasets/CDHD.yaml) which ensure that the dataset is used during the training process.

### 💾Environment
Our environment: Ubuntu 20.04, CUDA 12.2, NVIDIA RTX 3090 GPU.

Use conda to create the conda environment and activate it:
```shell
conda env create --name your_env_name python=3.8
conda activate your_env_name
pip install ultralytics
```
### 📈Training
Check the path in [train.py](./train.py), and run it to train:
```shell
python train.py 
```
### 📝Validation
Check the path in [val.py](./val.py).

```shell
python val.py
```

### 👀Performance
We present the visualization results of different methods on the cross-domain heterogeneous dataset (CDHD) in the figure below. Yellow circles represent missed or misdirected ship targets.

![visual](images/fig2.png)

## Acknowledgement

The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics).

Thanks for the great implementations! 

## Citation

If our code or models help your work, please cite our paper:
```BibTeX
@ARTICLE{10757443,
  author={Dong, Jun and Feng, Jiewen and Tang, Xiaoyu},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={OptiSAR-Net: A Cross-Domain Ship Detection Method for Multi-Source Remote Sensing Data}, 
  year={2024},
  doi={10.1109/TGRS.2024.3502447}}
```

<!-- links -->
[license-shield]: https://img.shields.io/github/license/shaojintian/Best_README_template.svg?style=flat-square
[license-url]: https://github.com/shaojintian/Best_README_template/blob/master/LICENSE.txt




