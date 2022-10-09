# IMPACT
Iterative Model-Based Phase Aberration Correction and Tomography (IMPACT)

IMPACT is an improved framework for distributed aberration correction in medical ultrasound imaging that relies on tomographic sound speed estimates. Our previous work (see https://github.com/rehmanali1994/DistributedAberrationCorrection) relies on coherence factor maximization to estimate a focusing sound speed at each imaging point. This focusing sound speed is then be used to reconstruct the spatial profile of local sound speed in the tissue. However, the performance of this coherence-based approach quickly breaks down in the presence of lateral variations in sound speed.

Instead, IMPACT uses aberration delays (or time shifts) measured between images from single-element transmissions to reconstruct the spatial profile of sound speed in the medium. The wavefield correlation technique (originally presented in https://github.com/rehmanali1994/DistributedAberrationCorrection) is used to correct aberrations in the image by using the estimated sound speed profile. However, rather than simply relying on a single set of measured aberration delays to reconstruct the sound speed profile, IMPACT iterates this process by using the latest sound speed estimate to reconstruct a set of better-focused ultrasound images, measure a new set of aberration delays from those images, and update the sound speed profile from the latest aberration delay measurements. 

The reconstruction of sound speed from aberration delay measurements is an extremely difficult inverse problem that is diagnostically relevant to medical ultrasound imaging. The primary motivation of making this work open sources is to demonstrate the principles behind tomographic sound speed estimation in a a more transparent manner so that other researchers can easily reproduce our work and improve upon it. The sample data and algorithms provided in this respository were used in following work:

> Ali, R.; Mitcham, T.; Singh, M.; Doyley, M.; Bouchard, R; Dahl, J; Duric, N. "Sound Speed Estimation for Distributed Aberration Correction in Laterally Varying Media". IEEE Transactions on Computational Imaging. *In Review*

If you use the algorithms and/or datasets provided in this repository for your own research work, please cite the above paper.

# Code and Sample Datasets
