# Privacy-Preserving Multimodal Biometric Verification

This repository contains the implementation and experimental notebook for the **privacy-preserving multimodal biometric verification** project described in the accompanying final report.

The project studies biometric verification using **face, fingerprint, and iris** modalities, with privacy protection based on **cancellable biometric templates**, evaluation under **same-key and cross-key conditions**, **score-level multimodal fusion**, and **CKKS encrypted matching**.

## Project framework

The implemented experimental workflow is:

```text
Face / Fingerprint / Iris
          |
          v
   Feature Embeddings
          |
          v
   Normalization
          |
          v
 Cancellable Transformation
          |
          v
   Protected Templates
       /       \
      v         v
 Same-Key    Cross-Key
 Evaluation  Evaluation
      |          |
      +----+-----+
           |
           v
   Score-Level Fusion
           |
           v
 CKKS Encrypted Matching
           |
           v
 Security / Utility / Runtime Evaluation
```

The final report describes this as the main experimental framework: embedding extraction and normalization, cancellable transformation, protected-template matching, same-key/cross-key evaluation, score-level fusion, CKKS encrypted matching, and security/utility/runtime analysis.

## Repository structure

```text
.
├── README.md
├── VERIFICATION.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── 01_prepare_data.py
│   ├── 02_fingerprint.py
│   ├── 03_face.py
│   ├── 04_iris.py
│   ├── 05_ckks.py
│   ├── 06_fusion.py
│   ├── 07_security_analysis.py
│   └── 08_final_results.py
│
├── notebooks/
│   └── final_pipeline_kaggle.ipynb
│
├── results/
│   ├── figures/
│   └── tables/
│
├── data/
│   └── README.md
│
├── checkpoints/
│   └── README.md
│
└── docs/
    └── final_report.pdf
```

## What each code file contains

### Data preparation
`src/01_prepare_data.py`

Creates the subject-level experimental split and verification-pair preparation used by the experiments.

### Fingerprint
`src/02_fingerprint.py`

Contains fingerprint preprocessing/training/embedding and verification-related processing used in the project.

### Face
`src/03_face.py`

Contains the face embedding and verification pipeline used in the experiments.

### Iris
`src/04_iris.py`

Contains the iris preprocessing/training/embedding and verification pipeline.

### Cancellable templates + CKKS
`src/05_ckks.py`

Contains the cancellable-template transformation and the true CKKS encrypted matching experiments for the biometric modalities.

### Multimodal fusion
`src/06_fusion.py`

Contains plain-domain and encrypted-domain **score-level fusion**, including multimodal/trimodal fusion and fusion-weight evaluation.

### Security and revocability
`src/07_security_analysis.py`

Contains the **same-key, cross-key, revocability, unlinkability, irreversibility, and security–utility analysis** used in the experiments.

### Final result generation
`src/08_final_results.py`

Contains the final result/table/figure-generation logic used for summarizing the experiments.

## Notebook

`notebooks/final_pipeline_kaggle.ipynb` is the original consolidated Kaggle experiment notebook. It is retained so that the complete experimental workflow and execution history remain available alongside the separated source files.

## Results

The repository intentionally keeps the result directories clean:

- `results/figures/` — place the final figures used in the report here.
- `results/tables/` — place the final tables/results exported from the experiments here.

Only **actual project outputs** should be added. Do not create or alter numerical results only to match the report.

## Final report

The final project report is available at:

`docs/final_report.pdf`

## Dataset

The original biometric datasets are not redistributed in this repository. See `data/README.md` for the dataset information and usage notes.

## Reproducibility note

The experiments were executed in Kaggle. The source files therefore retain the `/kaggle/input` and `/kaggle/working` paths used by the executed project. To reproduce the experiments outside Kaggle, those paths need to be mapped to the corresponding local/project data locations.

## Verification

The intended verification chain is:

```text
Source code
    ->
Kaggle experiment notebook
    ->
Generated experimental outputs
    ->
Figures / tables in results/
    ->
Final report in docs/
```

The repository does not claim that files placed in `results/` are newly generated unless they are actual outputs from the project. This keeps the GitHub repository consistent with the executed experiment.
