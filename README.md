# Privacy-Preserving Multimodal Biometric Verification

This repository contains the implementation and experimental artifacts for an M.Tech research project on privacy-preserving multimodal biometric verification.

## Overview

The project investigates a multimodal biometric verification framework combining:

- Face biometrics
- Fingerprint biometrics
- Iris biometrics
- Cancellable biometric templates
- Score-level multimodal fusion
- CKKS homomorphic encryption
- Same-key and cross-key security evaluation

The framework is evaluated using a subject-disjoint virtual multimodal dataset constructed from public face, fingerprint, and iris datasets.

## Experimental Pipeline

Public Datasets  
↓  
Virtual Subject Alignment  
↓  
Face / Fingerprint / Iris Embeddings  
↓  
Cancellable Template Transformation  
↓  
Similarity Matching  
↓  
Score-Level Fusion  
↓  
CKKS Encrypted Matching  
↓  
Security and Performance Evaluation

## Final Evaluation

The evaluation includes:

- Equal Error Rate (EER)
- ROC-AUC
- Accuracy
- TAR at fixed FAR
- Same-key evaluation
- Cross-key evaluation
- Recovery-gain analysis
- CKKS latency analysis

## Technologies

Python, PyTorch, NumPy, Pandas, scikit-learn, Matplotlib, TenSEAL and Jupyter Notebook.

## Note

The original public datasets are not included in this repository. Users should obtain datasets from their respective official sources and comply with their licensing and usage conditions.
