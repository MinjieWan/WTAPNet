# WTAPNet: Wavelet Transform-based Augmented Perception Network for Infrared Small Target Detection
This repository contains the codes for our published paper "WTAPNet: Wavelet Transform-based Augmented Perception Network for Infrared Small Target Detection". 

We provide the dataset used in our work. You can download them by the following link:
[[dataset (Google Drive)]](https://drive.google.com/file/d/18f4e7z4eakeYKGI8Hl7zRRZD--GSCPws/view?usp=sharing)

## Abstract
Infrared (IR) small target detection plays a vital role in various applications of both civil and military areas, such as marine rescue, forest-fire prevention and precise guidance. However, challenges stemming from the small size of infrared targets and noise interference in complex backgrounds often limit the accuracy of IR small target detection algorithms. Owing to the rapid development of deep learning, numerous convolutional neural network (CNN)-based methods have emerged in recent years, but they are inevitable to encounter the risk of target loss in deep layers due to the use of pooling layers. To address this problem, we present a wavelet transform-based augmented perception network, namely WTAPNet, in this paper. First, an enhancement and enlarge (EE) module is designed to improve the network’s perceptual capability for IR small targets by magnifying image resolution and augmenting target features at the same time. Then, a discrete wavelet transform-based downsampling (DWTD) module and an inverse wavelet transformbased fusion (IWTF) module are proposed. These two modules collaboratively work to extract and fuse multi-scale features, which simultaneously reduces information loss. Finally, a bottomup path fusion strategy is exploited to highlight and preserve small target features, which associates the lowest-level with the highest-level features and interconnects predictions from different hierarchical levels. Experimental results on NUDT-SIRST dataset and SIRST dataset demonstrate the superiority of our WTAPNet over other state-of-the-art IR small target detection methods in terms of F1-measure, recall and other indicators.

![绘图1](https://github.com/user-attachments/assets/e8d71485-e13f-427b-a439-da0f9c1b4eec)

## Requirements
Please see the requirements.txt contained in this respository.

## Usage
- Download and release dataset.zip, and put the released folder in the data folder. 
- Setup the required parameters in the main scripts.
- Run train.py for training. Also, you can use the pre-trained model we provided.
- Run test.py for testing.

## Citation
If you use our code and dataset for research, please cite our paper:

He H, Wan M, Xu Y, et al. WTAPNet: Wavelet Transform-based Augmented Perception Network for Infrared Small Target Detection[J]. IEEE Transactions on Instrumentation and Measurement, 2024, 73: 1-17.
