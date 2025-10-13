# Upper Tropospheric Humidity Prediction using Depp-learning architectures

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)
[![Status](https://img.shields.io/badge/status-production-success.svg)](https://github.com/BYU-Hydroinformatics/uth-prediction-unetpp-vae)

> **Advanced deep learning system for atmospheric humidity prediction through multi-satellite data fusion**

A state-of-the-art implementation of UNet++ with Variational Autoencoder bottleneck for gap-filling and prediction of Upper Tropospheric Humidity (UTH) using multi-satellite microwave radiometry. Supports dual operational modes: SAPHIR-enhanced accuracy in tropical/subtropical regions (±30° latitude) and global coverage using NOAA/EUMETSAT microwave sensors (1999-2021).
This repository presents a production-ready implementation of a deep learning pipeline 
for Upper Tropospheric Humidity (UTH) prediction using an advanced UNet++ architecture 
with Variational Autoencoder (VAE) bottleneck for uncertainty quantification. The system 
employs quality-weighted multi-satellite data fusion from up to eight NOAA and EUMETSAT 
microwave sensors (AMSU-B and MHS), with optional integration of SAPHIR relative humidity 
data from the Megha-Tropiques mission for enhanced accuracy within the ±30° latitude band.

The implementation provides two operational variants: (1) SAPHIR-enhanced mode with 
priority weighting for tropical and subtropical regions, and (2) microwave-only mode for 
global coverage spanning 1999-2021. The model architecture incorporates dense skip 
connections, squeeze-and-excitation attention blocks, satellite-specific normalization, 
and a variational bottleneck for probabilistic predictions with uncertainty estimates.

Key features include Docker containerization for reproducible deployment, mixed-precision 
training with automatic memory optimization, comprehensive loss functions (NLL, Charbonnier, 
KL divergence, gradient smoothness), and production-ready inference pipelines with temporal 
post-processing. The system is designed for both research applications requiring highest 
accuracy and operational use requiring global coverage.

Developed in collaboration with the National Remote Sensing Centre (NRSC), India, and 
BYU Hydroinformatics research group.


**Developed by:** Atmospheric Remote Sensing Research Team  
**In collaboration with:** National Remote Sensing Centre (NRSC), India & BYU Hydroinformatics  
**Contact:** saishashank3000@gmail.com, kvsm2k@gmail.com
