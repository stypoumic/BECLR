# Codebase of BECLR
### Official Implementation for ["BECLR: Batch Enhanced Contrastive Few-Shot Learning"](https://openreview.net/forum?id=k9SVcrmXL8)<br> (Stylianos Poulakakis-Daktylidis, Hadi Jamali-Rad)

* Accepted for **Spotlight** Presentation in ICLR 2024 
---

<!-- PapersWithCode-style SOTA badges for BECLR -->

[![PWC](https://img.shields.io/badge/Papers%20with%20Code-SOTA-success?style=flat-square&logo=read-the-docs)](#unsupervised-few-shot-learning-benchmark)
[![miniImageNet 1-shot](https://img.shields.io/badge/miniImageNet-5--way%201--shot%20%7C%2080.57%25-blue?style=flat-square)](#unsupervised-few-shot-learning-benchmark)
[![Task](https://img.shields.io/badge/task-unsupervised%20few--shot%20image%20classification-orange?style=flat-square)](#unsupervised-few-shot-learning-benchmark)

[![PWC](https://img.shields.io/badge/Papers%20with%20Code-SOTA-success?style=flat-square&logo=read-the-docs)](#unsupervised-few-shot-learning-benchmark)
[![miniImageNet 5-shot](https://img.shields.io/badge/miniImageNet-5--way%205--shot%20%7C%2087.82%25-blue?style=flat-square)](#unsupervised-few-shot-learning-benchmark)
[![Task](https://img.shields.io/badge/task-unsupervised%20few--shot%20image%20classification-orange?style=flat-square)](#unsupervised-few-shot-learning-benchmark)

[![PWC](https://img.shields.io/badge/Papers%20with%20Code-SOTA-success?style=flat-square&logo=read-the-docs)](#unsupervised-few-shot-learning-benchmark)
[![tieredImageNet 1-shot](https://img.shields.io/badge/tieredImageNet-5--way%201--shot%20%7C%2081.69%25-blueviolet?style=flat-square)](#unsupervised-few-shot-learning-benchmark)
[![Task](https://img.shields.io/badge/task-unsupervised%20few--shot%20image%20classification-orange?style=flat-square)](#unsupervised-few-shot-learning-benchmark)

[![PWC](https://img.shields.io/badge/Papers%20with%20Code-SOTA-success?style=flat-square&logo=read-the-docs)](#unsupervised-few-shot-learning-benchmark)
[![tieredImageNet 5-shot](https://img.shields.io/badge/tieredImageNet-5--way%205--shot%20%7C%2087.86%25-blueviolet?style=flat-square)](#unsupervised-few-shot-learning-benchmark)
[![Task](https://img.shields.io/badge/task-unsupervised%20few--shot%20image%20classification-orange?style=flat-square)](#unsupervised-few-shot-learning-benchmark)

[Benchmark](#unsupervised-few-shot-learning-benchmark) • 
[miniImageNet](#miniimagenet-5-way-classification) • 
[tieredImageNet](#tieredimagenet-5-way-classification) • 
[Notes](#notes)

---
## Abstract
Learning quickly from very few labeled samples is a fundamental attribute that separates machines and humans in the era of deep representation learning. Unsupervised few-shot learning (U-FSL) aspires to bridge this gap by discarding the reliance on annotations at training time. Intrigued by the success of contrastive learning approaches in the realm of U-FSL, we structurally approach their shortcomings in both pretraining and downstream inference stages. We propose a novel Dynamic Clustered mEmory (DyCE) module to promote a highly separable latent representation space for enhancing positive sampling at the pretraining phase and infusing implicit class-level insights into unsupervised contrastive learning. We then tackle the, somehow overlooked yet critical, issue of sample bias at the few-shot inference stage. We propose an iterative Optimal Transport-based distribution Alignment (OpTA) strategy and demonstrate that it efficiently addresses the problem, especially in low-shot scenarios where FSL approaches suffer the most from sample bias. We later on discuss that DyCE and OpTA are two intertwined pieces of a novel end-to-end approach (we coin as BECLR), constructively magnifying each other's impact. We then present a suite of extensive quantitative and qualitative experimentation to corroborate that BECLR sets a new state-of-the-art across ALL existing U-FSL benchmarks (to the best of our knowledge), and significantly outperforms the best of the current baselines, e.g. by up to $14$% and $12$% in the ($5$-way, $1$-shot) setting on miniImageNet and tieredImageNet, respectively.

<p align="center">
    <img src="images/beclr.png" width="85%" >
</p>

## Key Ideas
* **Going beyond instance-level contrastive learning.** In unsupervised contrastive FSL approaches each image within the batch and its augmentations correspond to a unique class, which is an unrealistic assumption. The pitfall here is that potential positive samples present within the same batch might then be repelled in the representation space, hampering the overall performance. We argue that infusing a semblance of class (or membership)-level insights into the unsupervised contrastive paradigm is essential. Our key idea to address this is extending the concept of memory queues by introducing inherent membership clusters represented by dynamically updated prototypes, while circumventing the need for large batch sizes. The proposed DyCE module facilitates a more meaningful positive sampling strategy by constructing and dynamically updating separable memory clusters.

* **Addressing inherent sample bias in (U-)FSL.** In Few-Shot learning the base (pretraining) and novel (inference) classes are mutually exclusive classes. This distribution shift poses a significant challenge at inference time for the swift adaptation to the novel classes. This is further aggravated due to access to only a few labeled (a.k.a support) samples within the few-shot task since the support samples are typically not representative of the larger unlabeled (a.k.a query) set. We refer to this phenomenon as sample bias, highlighting that it is overlooked by most (U-)FSL baselines. To address this issue, we introduce our OpTA add-on module within the supervised inference step. OpTA imposes no additional learnable parameters, yet efficiently aligns the representations of the labeled support and the unlabeled query sets, right before the final supervised inference step. We demonstrate that these two novel modules (DyCE and OpTA) are actually intertwined and amplify one another. Combining these two key ideas, we propose an end-to-end U-FSL approach coined as Batch-Enhanced Contrastive LeaRning (BECLR).


<p align="center">
  <img src="images/opta.png" width="85%" /> 
</p>

## Unsupervised Few-Shot Learning Benchmark

---

### miniImageNet (5-way classification)

| Method | Backbone | 1-shot | 5-shot |
|---|---|---:|---:|
| [C3LR](https://arxiv.org/abs/2202.08149) | Conv4 | 47.92 ± 1.20 | 64.81 ± 1.15 |
| [SAMPTransfer](https://arxiv.org/abs/2210.06339) | Conv4b | 61.02 ± 1.05 | 72.52 ± 0.68 |
| [LF2CS](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910406.pdf) | RN12 | 47.93 ± 0.19 | 66.44 ± 0.17 |
| [CPNWCP](https://dl.acm.org/doi/10.1007/978-3-031-19800-7_39) | RN18 | 53.14 ± 0.62 | 67.36 ± 0.50 |
| [SimCLR](https://proceedings.mlr.press/v119/chen20j.html) | RN18 | 62.58 ± 0.37 | 79.66 ± 0.27 |
| [SwAV†](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2) | RN18 | 59.84 ± 0.52 | 78.23 ± 0.26 |
| [NNCLR†](https://openaccess.thecvf.com/content/ICCV2021/html/Dwibedi_With_a_Little_Help_From_My_Friends_Nearest-Neighbor_Contrastive_Learning_ICCV_2021_paper.html) | RN18 | 63.33 ± 0.53 | 80.75 ± 0.25 |
| [SimSiam](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html) | RN18 | 62.80 ± 0.37 | 79.85 ± 0.27 |
| [HMS](https://ieeexplore.ieee.org/document/9786650) | RN18 | 58.20 ± 0.23 | 75.77 ± 0.16 |
| [Laplacian Eigenmaps](https://arxiv.org/abs/2210.03595) | RN18 | 59.47 ± 0.87 | 78.79 ± 0.58 |
| [PsCo†](https://arxiv.org/pdf/2303.00996) | RN18 | 47.24 ± 0.76 | 65.48 ± 0.68 |
| [UniSiam + dist](https://link.springer.com/chapter/10.1007/978-3-031-19800-7_43) | RN18 | 64.10 ± 0.36 | 82.26 ± 0.25 |
| [Meta-DM + UniSiam + dist∗](https://ieeexplore.ieee.org/document/10647300) | RN18 | 65.64 ± 0.36 | 83.97 ± 0.25 |
| **[BECLR (Ours)](https://arxiv.org/abs/2402.02444)** 🔥 | RN18 | **75.74 ± 0.62** | **84.93 ± 0.33** |
| **Deeper Backbone (RN50)** |  |  |  |
| [SwAV†](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2) | RN50 | 63.34 ± 0.42 | 82.76 ± 0.24 |
| [NNCLR†](https://openaccess.thecvf.com/content/ICCV2021/html/Dwibedi_With_a_Little_Help_From_My_Friends_Nearest-Neighbor_Contrastive_Learning_ICCV_2021_paper.html) | RN50 | 65.42 ± 0.44 | 83.31 ± 0.21 |
| [TrainProto](https://arxiv.org/abs/2106.10846) | RN50 | 58.92 ± 0.91 | 73.94 ± 0.63 |
| [UBC-FSL](https://openaccess.thecvf.com/content/CVPR2021W/LLID/html/Chen_Shot_in_the_Dark_Few-Shot_Learning_With_No_Base-Class_Labels_CVPRW_2021_paper.html) | RN50 | 56.20 ± 0.60 | 75.40 ± 0.40 |
| [PDA-Net](https://arxiv.org/abs/2105.11874) | RN50 | 63.84 ± 0.91 | 83.11 ± 0.56 |
| [UniSiam + dist](https://link.springer.com/chapter/10.1007/978-3-031-19800-7_43) | RN50 | 65.33 ± 0.36 | 83.22 ± 0.24 |
| [Meta-DM + UniSiam + dist∗](https://ieeexplore.ieee.org/abstract/document/10647300) | RN50 | 66.68 ± 0.36 | 85.29 ± 0.23 |
| **[BECLR (Ours)](https://arxiv.org/abs/2402.02444)** 🔥 | RN50 | **80.57 ± 0.57** | **87.82 ± 0.29** |

---

### tieredImageNet (5-way classification)

| Method | Backbone | 1-shot | 5-shot |
|---|---|---:|---:|
| [C3LR](https://arxiv.org/abs/2202.08149) | Conv4 | 42.37 ± 0.77 | 61.77 ± 0.25 |
| [SAMPTransfer](https://arxiv.org/abs/2210.06339) | Conv4b | 49.10 ± 0.94 | 65.19 ± 0.82 |
| [LF2CS](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910406.pdf) | RN12 | 53.16 ± 0.66 | 66.59 ± 0.57 |
| [CPNWCP](https://dl.acm.org/doi/10.1007/978-3-031-19800-7_39) | RN18 | 45.00 ± 0.19 | 62.96 ± 0.19 |
| [SimCLR](https://proceedings.mlr.press/v119/chen20j.html) | RN18 | 63.38 ± 0.42 | 79.17 ± 0.34 |
| [SwAV†](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2) | RN18 | 65.26 ± 0.53 | 81.73 ± 0.24 |
| [NNCLR†](https://openaccess.thecvf.com/content/ICCV2021/html/Dwibedi_With_a_Little_Help_From_My_Friends_Nearest-Neighbor_Contrastive_Learning_ICCV_2021_paper.html) | RN18 | 65.46 ± 0.55 | 81.40 ± 0.27 |
| [SimSiam](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html) | RN18 | 64.05 ± 0.40 | 81.40 ± 0.30 |
| [HMS](https://ieeexplore.ieee.org/document/9786650) | RN18 | 58.42 ± 0.25 | 75.85 ± 0.18 |
| [PsCo†](https://arxiv.org/pdf/2303.00996) | RN18 | 54.33 ± 0.54 | 69.73 ± 0.49 |
| [UniSiam + dist](https://link.springer.com/chapter/10.1007/978-3-031-19800-7_43) | RN18 | 67.01 ± 0.39 | 84.47 ± 0.28 |
| [Meta-DM + UniSiam + dist∗](https://ieeexplore.ieee.org/document/10647300) | RN18 | 67.11 ± 0.40 | 84.39 ± 0.28 |
| **[BECLR (Ours)](https://arxiv.org/abs/2402.02444)** 🔥 | RN18 | **76.44 ± 0.66** | **84.85 ± 0.37** |
| **Deeper Backbone (RN50)** |  |  |  |
| [SwAV†](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2) | RN50 | 68.02 ± 0.52 | 85.93 ± 0.33 |
| [NNCLR†](https://openaccess.thecvf.com/content/ICCV2021/html/Dwibedi_With_a_Little_Help_From_My_Friends_Nearest-Neighbor_Contrastive_Learning_ICCV_2021_paper.html) | RN50 | 69.82 ± 0.54 | 86.41 ± 0.31 |
| [UBC-FSL](https://openaccess.thecvf.com/content/CVPR2021W/LLID/html/Chen_Shot_in_the_Dark_Few-Shot_Learning_With_No_Base-Class_Labels_CVPRW_2021_paper.html) | RN50 | 66.60 ± 0.70 | 83.10 ± 0.50 |
| [PDA-Net](https://arxiv.org/abs/2105.11874) | RN50 | 69.01 ± 0.93 | 84.20 ± 0.69 |
| [UniSiam + dist](https://link.springer.com/chapter/10.1007/978-3-031-19800-7_43) | RN50 | 69.60 ± 0.38 | 86.51 ± 0.26 |
| [Meta-DM + UniSiam + dist∗](https://ieeexplore.ieee.org/abstract/document/10647300) | RN50 | 69.61 ± 0.38 | 86.53 ± 0.26 |
| **[BECLR (Ours)](https://arxiv.org/abs/2402.02444)** 🔥 | RN50 | **81.69 ± 0.61** | **87.86 ± 0.32** |

---

### Notes

- 🔥BECLR achieves state-of-the-art performance in this benchmark.
- † denotes reproduced results from prior work.
- ∗ indicates use of additional synthetic training data.
- All evaluations follow the standard 5-way episodic evaluation protocol. Refer to the paper for full experimental details and setup.

## Data Preparation
#### miniImageNet
* Download the miniImageNet dataset from [here](https://drive.google.com/file/d/1BfEBMlrf5UT4aNOoJPaa83CgbGWZAAAk/view?usp=sharing) (url courtesy of https://github.com/bbbdylan/unisiam/blob/main/README.md?plain=1) and unzip it within the ./data folder in the root directory of this project.
* Use the given split files to prepare the dataset:
```python
python dataset/create_split_miniImageNet.py --data_path "data/miniimagenet/miniimagenet" --split_path "dataset/split" --save_path "data/miniimagenet/miniimagenet_train" --partition "train"
```

#### tieredImageNet
* Download the ImageNet ILSVRC 2012 dataset from official website ([here](https://image-net.org/)) and unzip it within the ./data folder in the root directory of this project.
* Use the given split files to prepare the tieredImageNet dataset:
```python
python dataset/create_split_tieredImageNet.py --data_path "data/imagenet/train" --split_path "dataset/split" --save_path "data/imagenet/tieredimagenet_train" 
```

#### CIFAR-FS
* Download the CIFAR-FS dataset from Kaggle ([here](https://image-net.org/)) and unzip it within the ./data folder in the root directory of this project.

#### FC100
* Download the FC100 dataset from [here](https://image-net.org/) (url courtesy of https://github.com/kjunelee/MetaOptNet/tree/master) and unzip it within the ./data folder in the root directory of this project.

#### CDFSL
* Download the CDFSL dataset from the official CDFSL benchmark repository ([here](https://github.com/IBM/cdfsl-benchmark)) and unzip it within the ./data folder in the root directory of this project.

#### CUB
* Download the CUB dataset from [here](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view) and unzip it within the ./data folder in the root directory of this project.


Finally, the data directory should have the following structure:
```
\data
|___miniimagenet
|   |___miniimagenet
|   |   |___images
|   |___miniimagenet_train
|       |___0 ...
|       |___63
|   
|___imagenet
|   |___imagenet
|   |   |___train
|   |        |___n01440764 ...
|   |___tieredimagenet_train
|       |___0 ...
|       |___350
|
|___FC100
|   |___FC100_train.pickle
|   |___FC100_test.pickle
|   |___FC100_val.pickle
|
|___CIFAR-FS
|   |___CIFAR_FS_train.pickle
|   |___CIFAR_FS_test.pickle
|   |___CIFAR_FS_val.pickle
|
|___CUB_200_2011
|   |___CUB_200_2011
|       |___images ...
|
|___cdfsl
    |___chestX
    |   |___images ...
    |___EuroSAT
    |   |___2750
    |___ISIC
    |   |___ISIC2018_Task3_Training_GroundTruth
    |   |___ISIC2018_Task3_Training_Input
    |___plant-disease
        |___dataset
```

## Scripts for Training & Evaluation
First create an anaconda environment with all the required libraries, packages and dependencies: `conda env create -n beclr_env -f environment.yml`. Then use the following commands for running the training and evaluation scripts.
```python
python -m torch.distributed.launch train.py --cnfg_path "PATH_TO_TRAIN_CONFIG.JSON"
python -m torch.distributed.launch evaluate.py --cnfg_path "PATH_TO_EVAL_CONFIG.JSON"
```
Different configuration `json` files can be found in the \configs directory.

## Contact
Corresponding author: Stylianos Poulakakis-Daktylidis (<stypoumic@gmail.com>)


## Citation
```(bibtex)
@inproceedings{
poulakakis-daktylidis2024beclr,
title={{BECLR}: Batch Enhanced Contrastive Unsupervised Few-Shot Learning},
author={Stylianos Poulakakis-Daktylidis and Hadi Jamali-Rad},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=k9SVcrmXL8}
}
```
