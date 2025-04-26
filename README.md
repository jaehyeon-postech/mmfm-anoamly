<p align="center" width="100%">
<img src="figs/ov_logo.png"  width="60%" height="60%">
</p>

# Anomaly-OV (CVPR 2025 Highlight)
This is the official repository for our recent paper "Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models".
[![arXiv](https://img.shields.io/badge/arXiv-2502.07601-red.svg)](https://arxiv.org/abs/2502.07601)
[![Project Page](https://img.shields.io/badge/Project-Website-purple.svg)](https://xujiacong.github.io/Anomaly-OV/)

## Release Notes
- **[2025/04/25] 🔥 Anomaly-OV** is released and open to access. The implementation code and the contributed visual instruction tuning dataset & benchmark can be downloaded now. Please
remember to cite the source papers of the raw datasets.

## Models & Scripts

### Installation

#### 1. **Clone this repository and navigate to the Anomaly-OneVision folder:**
```bash
git clone https://github.com/honda-research-institute/Anomaly-OneVision.git
cd Anomaly-OneVision
```

#### 2. **Install the inference package:**
```bash
conda create -n anomaly_ov python=3.10 -y
conda activate anomaly_ov
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```
