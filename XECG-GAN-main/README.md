# XRay-GAN: Synthetic X-ray Image Generation using GANs

## Overview

This project explores the use of **Generative Adversarial Networks (GANs)** for generating synthetic X-ray images to augment medical imaging datasets.  
The objective is to enhance the diversity and balance of limited medical datasets, thereby improving the performance and generalization of deep learning models.

---

## Implemented Architectures

- Classical GAN  
- Deep Convolutional GAN (DCGAN)  
- Wasserstein GAN (WGAN)  
- Variational Autoencoder GAN (VAE-GAN)  
- BiLSTM-DCGAN (Experimental)

---

## Project Structure

XRay-GAN/
│
├── X-Rays GAN/
│ ├── CLASSICALGAN.ipynb
│ ├── DCGAN.ipynb
│ ├── WGAN.ipynb
│ ├── VAEDCGAN.ipynb
│ └── BiLSTMGAN.ipynb
│
└── README.md

Each notebook includes data preprocessing, model training, and synthetic X-ray image generation workflows.

---

## Requirements

- Python 3.8+  
- PyTorch  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- scikit-learn  

---

## Installation

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn

Usage
To train a GAN model, open the corresponding Jupyter notebook in the X-Rays GAN directory and execute the cells sequentially.

For example, to train a DCGAN model:
X-Rays GAN/DCGAN.ipynb

Results

DCGAN produced stable and visually realistic images.
WGAN improved convergence and reduced mode collapse.
Synthetic images demonstrate strong potential for augmenting X-ray datasets.

Future Work

Integrate explainable AI techniques (e.g., SHAP, LIME).
Quantitatively evaluate performance using metrics such as FID, SSIM, and PSNR.
Extend experiments to other medical modalities such as ECG and CT scans.

---
