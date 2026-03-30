# Tri-State Cyber-Physical Safety Architecture

[![DOI - Dataset & Code](https://zenodo.org/badge/DOI/10.5281/zenodo.YOUR_DOI_HERE.svg)](https://doi.org/10.5281/zenodo.YOUR_DOI_HERE)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Official repository for the research paper: **"A Unified Edge Architecture for Construction Safety: Expert-Calibrated Spatial Fusion, REBA Ergonomics, and Automated Incident Response"** by Daksh Singla, Samayank Goel, Sarthak Vishal Luhadia, and Logeswari G.

## 📖 Abstract
The construction industry remains disproportionately affected by fatal accidents and ergonomic injuries due to the inherent limitations of manual safety auditing. To address these critical gaps, this research proposes a unified, zero-latency cyber-physical system driven by a novel Tri-State Routing Engine. 

By engineering a deterministically stratified dataset, the system dynamically routes inference through a hardware-aware Hybrid Ensemble of YOLO11s and YOLO11m backbones. Overlapping spatial predictions are mathematically merged using a custom **Expert-Calibrated Weighted Boxes Fusion (EC-WBF)** algorithm, achieving a peak mAP@50 of 0.910. Crucially, the system bridges passive spatial detection with active kinematic tracking by routing dynamically cropped worker tensors to a lightweight `YOLO11n-pose` network, enabling simultaneous multi-target **REBA (Rapid Entire Body Assessment)** ergonomic tracking. Finally, the architecture integrates a natively coded, asynchronous Robotic Process Automation (RPA) daemon to compile Tier 1 life-safety alerts and Tier 2 administrative compliance digests.

## 🌟 Key Architectural Features
* **Tri-State Routing Engine:** Dynamically routes inference between YOLO11s and YOLO11m models based on local VRAM hardware constraints.
* **EC-WBF Algorithm:** Custom Expert-Calibrated Weighted Boxes Fusion matrix that mathematically merges overlapping tensors to eliminate semantic hallucinations.
* **Spatiotemporal REBA Tracking:** Full-body kinematic tracking to calculate trunk, neck, shoulder, and knee flexion angles to mathematically prevent ergonomic strain.
* **Implicit Negation Geometry:** Mathematical failsafes that project penalty zones for missing PPE even under severe visual occlusion.
* **Asynchronous RPA Matrix:** A decoupled SMTP daemon that fires zero-latency life-safety alerts (Falls) and generates automated Shift-End PDF Audit digests without dropping GPU inference frames.

## 📂 Data & Weights Availability
To ensure full scientific reproducibility, all datasets and pre-trained models are open-source:
* **Dataset:** The mathematically stratified multi-hazard dataset (V1 Baseline through V4 Stratified) is hosted on Zenodo: `[INSERT ZENODO DATASET LINK HERE]`
* **Model Weights:** The trained `.pt` weights for the Spatial Expert (Small), Micro/Context Experts (Medium), and Pose models are available in the **[Releases](https://github.com/YOUR_USERNAME/Tri-State-Construction-Safety-CV/releases)** section of this repository.

## ⚙️ Hardware Requirements
* **Minimum Edge Deployment:** Intel Core i7, NVIDIA GTX 1650 Ti (4GB VRAM) -> *Yields ~14.2 FPS on Hybrid Mode.*
* **Enterprise Cloud Deployment:** NVIDIA A40 (48GB VRAM) -> *Yields ~124.1 FPS on Hybrid Mode.*

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Tri-State-Construction-Safety-CV.git](https://github.com/YOUR_USERNAME/Tri-State-Construction-Safety-CV.git)
   cd Tri-State-Construction-Safety-CV
