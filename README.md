# IMPACT
Iterative Model-Based Phase Aberration Correction and Tomography (IMPACT)

IMPACT is a new framework for sound speed estimation and distributed aberration correction in medical ultrasound imaging. Our previous work (see https://github.com/rehmanali1994/DistributedAberrationCorrection) relies on coherence factor maximization to estimate a focusing sound speed at each imaging point. This focusing sound speed is then used to reconstruct the spatial profile of local sound speed in the tissue. However, the performance of this approach quickly breaks down in the presence of lateral variations in the sound speed profile.

Instead, IMPACT uses aberration delays (or time shifts) measured between images from single-element transmissions to reconstruct the spatial profile of sound speed in the medium. The wavefield correlation technique (originally presented in https://github.com/rehmanali1994/DistributedAberrationCorrection) is used to correct aberrations in the image by using the estimated sound speed profile. However, rather than simply relying on a single set of measured aberration delays to reconstruct the sound speed profile, IMPACT iterates this process by using the latest sound speed estimate to reconstruct a set of newly focused ultrasound images, measure a new set of aberration delays from those images, and update the sound speed profile from the latest aberration delay measurements. 

The reconstruction of sound speed from aberration delays is an extremely difficult inverse problem of key diagnostic relevance to medical ultrasound imaging. The primary motivation of this open-source work is to demonstrate the principles behind sound speed tomography in a more transparent manner so that other researchers can easily reproduce our work and improve upon it. The sample data and algorithms provided in this respository were used in following work:

> Ali, R.; Mitcham, T.; Singh, M.; Doyley, M.; Bouchard, R.; Dahl, J.; Duric, N. "Sound Speed Estimation for Distributed Aberration Correction in Laterally Varying Media". IEEE Transactions on Computational Imaging. *In Review*

If you use the algorithms and/or datasets provided in this repository for your own research work, please cite the above paper.

You can reference a static version of this code by its DOI number: [![DOI](https://zenodo.org/badge/548574417.svg)](https://zenodo.org/badge/latestdoi/548574417)

# Code and Sample Datasets

**Please download the sample data (AbdominalMap3.mat; AbdominalMap4.mat; RatAbdomenL12-3v.mat; PhantomL12-5-50mm.mat) under the [releases](https://github.com/rehmanali1994/IMPACT/releases) tab for this repository, and place that data in the ([Datasets](https://github.com/rehmanali1994/IMPACT/tree/main/Datasets/)) folder.**

The following scripts correspond to each dataset:
1) AbdominalMap3.mat and AbdominalMap4.mat - [AberrationTomographyShotGatherMigKWave.m](https://github.com/rehmanali1994/IMPACT/blob/main/MATLAB/AberrationTomographyShotGatherMigKWave.m)) and [AberrationTomographyShotGatherMigKWave.py](https://github.com/rehmanali1994/IMPACT/blob/main/Python/AberrationTomographyShotGatherMigKWave.py)) - These datasets were simulated in k-Wave with a known ground-truth sound speed profile.
2) RatAbdomenL12-3v.mat - [AberrationTomographyShotGatherMigL12_3v.m](https://github.com/rehmanali1994/IMPACT/blob/main/MATLAB/AberrationTomographyShotGatherMigL12_3v.m)) and [AberrationTomographyShotGatherMigL12_3v.py](https://github.com/rehmanali1994/IMPACT/blob/main/Python/AberrationTomographyShotGatherMigL12_3v.py)) - This dataset was obtained from the abdomen of an obese Zucker rat under a Stanford-approved IACUC protocol using an L12-3v probe.
3) PhantomL12-5-50mm.mat - [AberrationTomographyShotGatherMigL12_5_50mm.m](https://github.com/rehmanali1994/IMPACT/blob/main/MATLAB/AberrationTomographyShotGatherMigL12_5_50mm.m)) and [AberrationTomographyShotGatherMigL12_5_50mm.py](https://github.com/rehmanali1994/IMPACT/blob/main/Python/AberrationTomographyShotGatherMigL12_5_50mm.py)) - This dataset was obtained using an L12-5 50mm probe from a phantom with 3 high sound speed alcohol inclusions to induce aberrations in the image.

The following key MATLAB functions (implemented in Python within [functions.py](https://github.com/rehmanali1994/IMPACT/blob/main/Python/functions.py)) used in these scripts are: 
1) [line_pixel_intersection.m](https://github.com/rehmanali1994/IMPACT/tree/main/MATLAB/functions/line_pixel_intersection.m) - Computation of path length over each pixel in sound speed map for travel-time tomography
2) [optApod.m](https://github.com/rehmanali1994/IMPACT/tree/main/MATLAB/functions/optApod.m) - Computes the optimal apodization that maximizes the short-lag spatial coherence. If using this script, please cite the following work:
> Ali, R.; Duric, N.; Dahl, J. "Optimal Transmit Apodization for the Maximization of Lag-One Coherence with Applications to Aberration Delay Estimation". Ultrasonics. *In Review*
3) [propagate.m](https://github.com/rehmanali1994/IMPACT/tree/main/MATLAB/functions/propagate.m) - The Fourier split-step angular spectrum method used in the wavefield correlation technique
4) [spray.m](https://github.com/rehmanali1994/IMPACT/tree/main/MATLAB/functions/spray.m) - Adjoint of masking operation used to select imaging points in the medium

Overall, we strongly recommend and prefer the MATLAB code over the Python code because of its faster run times. We also recommend running this code on a system with a large amount of RAM (at least 16 GB, but 32-64 GB is ideal).

# Sample Results
1) AbdominalMap3.mat and AbdominalMap4.mat:

![](https://github.com/rehmanali1994/IMPACT/blob/main/Python/figures/AbdominalMaps.png)

2) RatAbdomenL12-3v.mat

![](https://github.com/rehmanali1994/IMPACT/blob/main/Python/figures/RatAbdomenL12-3v.png)

3) PhantomL12-5-50mm.mat

![](https://github.com/rehmanali1994/IMPACT/blob/main/Python/figures/PhantomL12-5-50mm.png)
