# Verification Guide

A reviewer can verify the project in the following order:

1. Read `README.md` for the complete framework.
2. Open `notebooks/final_pipeline_kaggle.ipynb` to inspect the consolidated experiment.
3. Inspect the separated source files in `src/`.
4. Compare the figures and tables in `results/` with the corresponding material in `docs/final_report.pdf`.
5. Follow the experiment outputs back to the corresponding code/notebook sections.

The main experimental components represented in the repository are:

- face, fingerprint, and iris processing;
- biometric embeddings;
- cancellable biometric transformation;
- same-key evaluation;
- cross-key/revocability/unlinkability evaluation;
- multimodal and trimodal score-level fusion;
- CKKS encrypted matching;
- security, utility, and runtime analysis.

The original execution environment was Kaggle, so the notebook/source files contain Kaggle-specific paths.
