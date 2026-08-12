# A System-Level Security–Utility Evaluation Framework for Privacy-Preserving Multimodal Biometric Verification Using Cancellable Templates and CKKS Homomorphic Encryption

## Project Overview

This project presents a system-level security–utility evaluation framework for privacy-preserving multimodal biometric verification using face, fingerprint, and iris modalities.

The work extends a baseline multimodal biometric verification framework by incorporating:

- Cancellable biometric template transformation
- Same-key protected verification
- Cross-key evaluation for practical unlinkability analysis
- Protected score-level fusion
- CKKS-based encrypted matching
- Recovery-gain analysis
- Latency and security–utility analysis

The objective is not to propose a new biometric recognition model, cancellable transformation algorithm, or cryptographic scheme. Instead, the project evaluates how these components interact within a common multimodal verification protocol.

---

## Research Objectives

The main objectives are:

1. Evaluate the effect of cancellable template transformation on face, fingerprint, and iris verification.
2. Compare binary and non-binary protected representations across projection dimensions.
3. Evaluate protected-template utility using same-key verification.
4. Analyse cross-key behaviour as empirical evidence of practical unlinkability and renewability.
5. Evaluate protected score-level fusion as a mechanism for recovering utility after template protection.
6. Examine whether CKKS encrypted matching preserves protected-domain verification behaviour.
7. Analyse the overall security–utility and latency trade-off.

---

## Dataset

A virtually subject-aligned multimodal dataset was constructed from three public biometric datasets:

- **VGGFace2** — Face
- **SOCOFing-real** — Fingerprint
- **CASIA-Iris-Thousand** — Iris

The resulting dataset contains:

- 600 virtual subjects
- 10 samples per modality per subject
- 18,000 biometric samples in total
- 480 training subjects
- 60 validation subjects
- 60 testing subjects

A subject-level split was used to avoid identity overlap between training, validation, and testing.

For validation and testing, balanced verification sets containing 2,700 genuine and 2,700 impostor comparisons were generated, giving 5,400 comparisons per split.

---

## System Pipeline

The overall pipeline consists of:

1. Multimodal dataset preparation
2. Subject-level train/validation/test splitting
3. Balanced verification-pair generation
4. Deep embedding extraction
5. Embedding normalization
6. Cancellable template transformation
7. Same-key and cross-key evaluation
8. Protected score normalization and fusion
9. CKKS encrypted-domain matching
10. Security, utility, and latency evaluation

---

## Feature Extraction

Different deep feature extractors are used for the three modalities:

| Modality | Feature Extractor | Representation |
|---|---|---|
| Face | InceptionResnetV1 pretrained on VGGFace2 | 512-D embedding |
| Fingerprint | ResNet-based fingerprint model | 512-D embedding |
| Iris | EfficientNet-B3 | 512-D embedding |

The same embedding representations are used across plain, protected, fused, and CKKS evaluation settings.

---

## Cancellable Template Protection

Cancellable protection is implemented using key-dependent random projection.

The experiments evaluate:

- Projection dimensions: 64, 128, 256, 512, and 1024
- Binary and non-binary protected representations
- Same-key verification
- Cross-key verification

Same-key evaluation measures retained verification utility.

Cross-key evaluation is used as empirical evidence for practical unlinkability and renewability. It is not treated as a formal mathematical proof of irreversibility or unconditional unlinkability.

---

## Multimodal Fusion

Protected modality scores are normalized and combined using validation-selected score-level fusion.

Both bimodal and trimodal configurations are evaluated:

- Face + Fingerprint
- Face + Iris
- Fingerprint + Iris
- Face + Fingerprint + Iris

Fusion is analysed not only as an accuracy-improvement mechanism but also as a potential utility-recovery mechanism after cancellable transformation.

---

## CKKS Encrypted Matching

CKKS homomorphic encryption is used to evaluate encrypted similarity computation over protected biometric representations.

The implementation uses **TenSEAL**.

Main CKKS configuration:

```text
Scheme: CKKS
poly_modulus_degree: 8192
coeff_mod_bit_sizes: [60, 40, 40, 60]
global_scale: 2^40
Operation: Encrypted inner-product / similarity computation
