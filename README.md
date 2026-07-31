# S²RE-Net: A Semantic Response Enhancement Network for Scribble-Supervised Cardiac MRI Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📌 Overview

**S²RE-Net** is a state-of-the-art weakly supervised segmentation framework for cardiac MRI using **scribble annotations** only. Our method addresses the critical challenges of sparse supervision—response drift, boundary discontinuities, and missed thin-walled structures—through a novel three-expert collaborative architecture with dynamic fusion and closed-loop optimization.

This repository contains the official PyTorch implementation of the paper:

> **S²RE-Net: A Semantic Response Enhancement Network for Scribble-Supervised Cardiac MRI Segmentation**  
> *Md Abu Sufian, Mingbo Niu, et al.*  
> IEEE Transactions on Medical Imaging (TMI), 2025

---

## 🔥 Key Features

- ✅ **Three-Expert Collaborative Framework** – Transformer (global context) + CNN (local details) + SR-Branch (structural recovery)
- ✅ **Dynamic Mix** – Pixel-wise adaptive fusion with reverse pseudo-label closed-loop constraints
- ✅ **HMNA Block** – High-Order Multi-Scale Non-Local Attention for multi-scale contextual modeling
- ✅ **Scribble-Supervised** – Trains with only sparse scribble annotations, significantly reducing labeling cost
- ✅ **State-of-the-Art Performance** – Outperforms 17 methods on ACDC and MSCMRseg datasets
- ✅ **Fully Reproducible** – Includes code, configs, and evaluation scripts

---

## 📊 Results

### ACDC Dataset

| Method | Dice (RV) | Dice (MYO) | Dice (LV) | Dice (Avg) | Vol. Error ↓ |
|--------|-----------|------------|-----------|------------|--------------|
| S²RE-Net (Ours) | **0.8451** | **0.8468** | **0.9231** | **0.8717** | **0.1889** |
| CycleMix | 0.8266 | 0.7595 | 0.8269 | 0.8269 | 0.2446 |
| FDDSeg | 0.7984 | 0.7987 | 0.8831 | 0.8267 | 0.2011 |

### MSCMRseg Dataset

| Method | Dice (RV) | Dice (MYO) | Dice (LV) | Dice (Avg) | Vol. Error ↓ |
|--------|-----------|------------|-----------|------------|--------------|
| S²RE-Net (Ours) | **0.8547** | **0.7904** | **0.8978** | **0.8476** | **0.1073** |
| CycleMix | 0.7507 | 0.8162 | 0.8889 | 0.8186 | 0.2679 |
| FDDSeg | 0.7798 | 0.7039 | 0.8878 | 0.7905 | 0.2675 |

---

## 🏗️ Architecture
