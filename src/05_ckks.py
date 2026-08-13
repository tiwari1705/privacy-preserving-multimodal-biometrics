"""Extracted from notebook3e7611700a(1).ipynb.
Cells preserved in original execution order.
Kaggle paths are intentionally preserved; replace them for local execution.
"""

# ================= NOTEBOOK CELL 32 =================
# ============================================================
# CKKS ENCRYPTED MATCHING FOR FINGERPRINT EMBEDDINGS
# - Uses TenSEAL CKKS
# - Encrypted-domain dot-product matching
# - Decrypts scores
# - Evaluates biometric metrics
# ============================================================

# If needed on Kaggle / Colab:
# !pip install tenseal

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. IMPORT TENSEAL
# ============================================================
import tenseal as ts

# ============================================================
# 2. PATHS
# ============================================================
EMB_DIR = "/kaggle/working/fingerprint_balanced_test_outputs"
PAIR_DIR = "/kaggle/working/common_pairs_balanced"
SAVE_DIR = "/kaggle/working/fingerprint_ckks_outputs"

os.makedirs(SAVE_DIR, exist_ok=True)

TEST_EMB_PATH = os.path.join(EMB_DIR, "fingerprint_test_embeddings.npy")
TEST_META_PATH = os.path.join(EMB_DIR, "fingerprint_test_embeddings_meta.csv")
PAIR_CSV = os.path.join(PAIR_DIR, "test_pairs_common_balanced.csv")

# ============================================================
# 3. HELPERS
# ============================================================
def clean_sid(x):
    return str(int(float(x))).zfill(3)

def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    return float(eer), float(threshold), fpr, tpr, thresholds

def tar_at_far(labels, scores, target_far=0.01):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmin(np.abs(fpr - target_far))
    return float(tpr[idx]), float(fpr[idx]), float(thresholds[idx])

# ============================================================
# 4. LOAD EMBEDDINGS + META + PAIRS
# ============================================================
test_embs = np.load(TEST_EMB_PATH)
test_meta = pd.read_csv(TEST_META_PATH)
pair_df = pd.read_csv(PAIR_CSV)

test_meta["subject"] = test_meta["subject"].apply(clean_sid)
pair_df["subject1"] = pair_df["subject1"].apply(clean_sid)
pair_df["subject2"] = pair_df["subject2"].apply(clean_sid)

print("Embeddings shape:", test_embs.shape)
print("Meta shape      :", test_meta.shape)
print("Pairs shape     :", pair_df.shape)

# ============================================================
# 5. BUILD EMBEDDING MAP
# ============================================================
emb_map = {}
for i, row in test_meta.iterrows():
    emb_map[(row["subject"], int(row["img_idx"]))] = test_embs[i].astype(np.float64)

print("Embedding map size:", len(emb_map))

# ============================================================
# 6. CKKS CONTEXT
# ============================================================
# CKKS practical setup for ~512-d embeddings
# poly_modulus_degree must be large enough for slot count and noise budget
# coeff_mod_bit_sizes example commonly used in practice

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

# We need secret key for decryption in this demo.
# In a real deployment, the matching server would typically get
# only the public context / serialized public part.
secret_context = context
public_context = context.copy()
public_context.make_context_public()

print("CKKS context ready")

# ============================================================
# 7. OPTIONAL: ENCRYPT ALL TEST EMBEDDINGS FIRST
# ============================================================
# Since embeddings are already L2-normalized from your model,
# cosine similarity = dot product.

encrypted_emb_map = {}

start_encrypt = time.time()

for key, vec in emb_map.items():
    encrypted_emb_map[key] = ts.ckks_vector(public_context, vec.tolist())

encrypt_time = time.time() - start_encrypt
print(f"Encrypted {len(encrypted_emb_map)} embeddings in {encrypt_time:.2f} sec")

# ============================================================
# 8. ENCRYPTED MATCHING
# ============================================================
# Strategy:
# Encrypt e1 under CKKS
# Dot with plaintext e2 inside HE object
# Decrypt scalar score

scores = []
labels = []
valid_rows = []

start_match = time.time()

for _, row in pair_df.iterrows():
    pair_id = int(row["pair_id"])
    s1 = row["subject1"]
    s2 = row["subject2"]
    idx1 = int(row["idx1"])
    idx2 = int(row["idx2"])
    label = int(row["label"])

    k1 = (s1, idx1)
    k2 = (s2, idx2)

    if k1 not in encrypted_emb_map or k2 not in emb_map:
        continue

    enc_e1 = encrypted_emb_map[k1]
    e2 = emb_map[k2]

    # encrypted dot product
    enc_score = enc_e1.dot(e2.tolist())

    # decrypt scalar
    dec_score = enc_score.decrypt(secret_context.secret_key())

    # dec_score may be scalar or length-1 list depending on backend behavior
    if isinstance(dec_score, list):
        score = float(dec_score[0])
    else:
        score = float(dec_score)

    scores.append(score)
    labels.append(label)
    valid_rows.append([pair_id, s1, s2, idx1, idx2, label, score])

match_time = time.time() - start_match

score_df = pd.DataFrame(
    valid_rows,
    columns=["pair_id", "subject1", "subject2", "idx1", "idx2", "label", "score"]
)

score_csv_path = os.path.join(SAVE_DIR, "fingerprint_test_scores_ckks.csv")
score_df.to_csv(score_csv_path, index=False)

scores = np.array(scores, dtype=np.float64)
labels = np.array(labels, dtype=np.int32)

print(f"Pairs evaluated: {len(score_df)}")
print(f"Encrypted matching time: {match_time:.2f} sec")

if len(scores) == 0:
    raise ValueError("No valid encrypted scores were computed.")

# ============================================================
# 9. EVALUATION
# ============================================================
eer, threshold, fpr, tpr, thresholds = compute_eer(scores, labels)
roc_auc = auc(fpr, tpr)

preds = (scores >= threshold).astype(int)

acc = accuracy_score(labels, preds)
prec = precision_score(labels, preds, zero_division=0)
rec = recall_score(labels, preds, zero_division=0)
f1 = f1_score(labels, preds, zero_division=0)

cm = confusion_matrix(labels, preds)
TN, FP, FN, TP = cm.ravel()

far = FP / (FP + TN)
frr = FN / (FN + TP)

tar1, actual_far1, thr_far1 = tar_at_far(labels, scores, target_far=0.01)
tar01, actual_far01, thr_far01 = tar_at_far(labels, scores, target_far=0.001)

metrics = {
    "scheme": "CKKS",
    "library": "TenSEAL",
    "num_pairs_evaluated": int(len(score_df)),
    "embedding_dim": int(test_embs.shape[1]),
    "encrypt_time_sec": float(encrypt_time),
    "matching_time_sec": float(match_time),
    "eer": float(eer),
    "threshold_at_eer": float(threshold),
    "roc_auc": float(roc_auc),
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "far_at_eer_threshold": float(far),
    "frr_at_eer_threshold": float(frr),
    "tar_at_far_1_percent": float(tar1),
    "actual_far_near_1_percent": float(actual_far1),
    "threshold_near_far_1_percent": float(thr_far1),
    "tar_at_far_0_1_percent": float(tar01),
    "actual_far_near_0_1_percent": float(actual_far01),
    "threshold_near_far_0_1_percent": float(thr_far01),
    "tp": int(TP),
    "tn": int(TN),
    "fp": int(FP),
    "fn": int(FN),
}

with open(os.path.join(SAVE_DIR, "fingerprint_ckks_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("\n🔥 CKKS ENCRYPTED MATCHING RESULTS")
print(f"EER              : {eer*100:.2f}%")
print(f"Threshold@EER    : {threshold:.4f}")
print(f"ROC-AUC          : {roc_auc:.4f}")
print(f"Accuracy         : {acc:.4f}")
print(f"Precision        : {prec:.4f}")
print(f"Recall           : {rec:.4f}")
print(f"F1-score         : {f1:.4f}")
print(f"FAR              : {far:.4f}")
print(f"FRR              : {frr:.4f}")
print(f"TAR@FAR=1%       : {tar1:.4f}")
print(f"TAR@FAR=0.1%     : {tar01:.4f}")

# ============================================================
# 10. SHOW ROC CURVE
# ============================================================
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Fingerprint CKKS ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Fingerprint CKKS ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fingerprint_ckks_roc_curve.png"), dpi=200)
plt.close()

# ============================================================
# 11. SHOW SCORE DISTRIBUTION
# ============================================================
genuine_scores = scores[labels == 1]
impostor_scores = scores[labels == 0]

plt.figure(figsize=(7, 5))
plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
plt.axvline(threshold, linestyle="--", label=f"Thr@EER={threshold:.4f}")
plt.xlabel("Decrypted CKKS Score")
plt.ylabel("Density")
plt.title("Fingerprint CKKS Score Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
plt.axvline(threshold, linestyle="--", label=f"Thr@EER={threshold:.4f}")
plt.xlabel("Decrypted CKKS Score")
plt.ylabel("Density")
plt.title("Fingerprint CKKS Score Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fingerprint_ckks_score_distribution.png"), dpi=200)
plt.close()

# ============================================================
# 12. CONFUSION MATRIX
# ============================================================
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Fingerprint CKKS Confusion Matrix")
plt.colorbar()
ticks = np.arange(2)
plt.xticks(ticks, ["Impostor", "Genuine"])
plt.yticks(ticks, ["Impostor", "Genuine"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Fingerprint CKKS Confusion Matrix")
plt.colorbar()
plt.xticks(ticks, ["Impostor", "Genuine"])
plt.yticks(ticks, ["Impostor", "Genuine"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fingerprint_ckks_confusion_matrix.png"), dpi=200)
plt.close()

# ============================================================
# 13. SAVE ROC POINTS + SUMMARY
# ============================================================
roc_df = pd.DataFrame({
    "fpr": fpr,
    "tpr": tpr,
    "threshold": thresholds
})
roc_df.to_csv(os.path.join(SAVE_DIR, "fingerprint_ckks_roc_points.csv"), index=False)

with open(os.path.join(SAVE_DIR, "fingerprint_ckks_summary.txt"), "w") as f:
    f.write("FINGERPRINT CKKS ENCRYPTED MATCHING SUMMARY\n")
    f.write("===========================================\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")

print("\nConfusion Matrix:")
print(cm)

print("\n✅ Saved files:")
for fn in sorted(os.listdir(SAVE_DIR)):
    print(os.path.join(SAVE_DIR, fn))

# ================= NOTEBOOK CELL 37 =================
# ============================================================
# FACE CKKS ENCRYPTED MATCHING + EVALUATION
# - Uses pretrained face embeddings
# - Encrypted-domain dot-product matching with CKKS
# - Decrypt scores
# - Evaluate metrics + curves
# ============================================================

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# =========================
# 1. IMPORT TENSEAL
# =========================
import tenseal as ts

# =========================
# 2. PATHS
# =========================
EMB_DIR = "/kaggle/working/face_pretrained_embeddings"
PAIR_DIR = "/kaggle/working/common_pairs_balanced"
SAVE_DIR = "/kaggle/working/face_ckks_outputs"

os.makedirs(SAVE_DIR, exist_ok=True)

TEST_EMB_PATH = os.path.join(EMB_DIR, "face_test_embeddings_pretrained.npy")
TEST_META_PATH = os.path.join(EMB_DIR, "face_test_embeddings_meta_pretrained.csv")
PAIR_CSV = os.path.join(PAIR_DIR, "test_pairs_common_balanced.csv")

# =========================
# 3. HELPERS
# =========================
def clean_sid(x):
    return str(int(float(x))).zfill(3)

def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    return float(eer), float(threshold), fpr, tpr, thresholds

def tar_at_far(labels, scores, target_far=0.01):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmin(np.abs(fpr - target_far))
    return float(tpr[idx]), float(fpr[idx]), float(thresholds[idx])

# =========================
# 4. LOAD EMBEDDINGS + META + PAIRS
# =========================
test_embs = np.load(TEST_EMB_PATH)
test_meta = pd.read_csv(TEST_META_PATH)
pair_df = pd.read_csv(PAIR_CSV)

test_meta["subject"] = test_meta["subject"].apply(clean_sid)
pair_df["subject1"] = pair_df["subject1"].apply(clean_sid)
pair_df["subject2"] = pair_df["subject2"].apply(clean_sid)

print("Embeddings shape:", test_embs.shape)
print("Meta shape      :", test_meta.shape)
print("Pairs shape     :", pair_df.shape)

# =========================
# 5. BUILD EMBEDDING MAP
# =========================
emb_map = {}
for i, row in test_meta.iterrows():
    emb_map[(row["subject"], int(row["img_idx"]))] = test_embs[i].astype(np.float64)

print("Embedding map size:", len(emb_map))

# =========================
# 6. CKKS CONTEXT
# =========================
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

secret_context = context
public_context = context.copy()
public_context.make_context_public()

print("CKKS context ready")

# =========================
# 7. ENCRYPT TEST EMBEDDINGS
# =========================
encrypted_emb_map = {}

start_encrypt = time.time()
for key, vec in emb_map.items():
    encrypted_emb_map[key] = ts.ckks_vector(public_context, vec.tolist())
encrypt_time = time.time() - start_encrypt

print(f"Encrypted {len(encrypted_emb_map)} embeddings in {encrypt_time:.2f} sec")

# =========================
# 8. ENCRYPTED MATCHING
# =========================
scores = []
labels = []
valid_rows = []

start_match = time.time()

for _, row in pair_df.iterrows():
    pair_id = int(row["pair_id"])
    s1 = row["subject1"]
    s2 = row["subject2"]
    idx1 = int(row["idx1"])
    idx2 = int(row["idx2"])
    label = int(row["label"])

    k1 = (s1, idx1)
    k2 = (s2, idx2)

    if k1 not in encrypted_emb_map or k2 not in emb_map:
        continue

    enc_e1 = encrypted_emb_map[k1]
    e2 = emb_map[k2]

    # encrypted dot product
    enc_score = enc_e1.dot(e2.tolist())

    # decrypt scalar
    dec_score = enc_score.decrypt(secret_context.secret_key())

    if isinstance(dec_score, list):
        score = float(dec_score[0])
    else:
        score = float(dec_score)

    scores.append(score)
    labels.append(label)
    valid_rows.append([pair_id, s1, s2, idx1, idx2, label, score])

match_time = time.time() - start_match

score_df = pd.DataFrame(
    valid_rows,
    columns=["pair_id", "subject1", "subject2", "idx1", "idx2", "label", "score"]
)

score_df.to_csv(os.path.join(SAVE_DIR, "face_test_scores_ckks.csv"), index=False)

scores = np.array(scores, dtype=np.float64)
labels = np.array(labels, dtype=np.int32)

print("Pairs requested :", len(pair_df))
print("Pairs evaluated :", len(score_df))
print(f"Encrypted matching time: {match_time:.2f} sec")

if len(scores) == 0:
    raise ValueError("No valid encrypted face scores were computed.")

# =========================
# 9. EVALUATION
# =========================
eer, threshold, fpr, tpr, thresholds = compute_eer(scores, labels)
roc_auc = auc(fpr, tpr)

preds = (scores >= threshold).astype(int)

acc = accuracy_score(labels, preds)
prec = precision_score(labels, preds, zero_division=0)
rec = recall_score(labels, preds, zero_division=0)
f1 = f1_score(labels, preds, zero_division=0)

cm = confusion_matrix(labels, preds)
TN, FP, FN, TP = cm.ravel()

far = FP / (FP + TN)
frr = FN / (FN + TP)

tar1, actual_far1, thr_far1 = tar_at_far(labels, scores, target_far=0.01)
tar01, actual_far01, thr_far01 = tar_at_far(labels, scores, target_far=0.001)

metrics = {
    "scheme": "CKKS",
    "library": "TenSEAL",
    "num_pairs_evaluated": int(len(score_df)),
    "embedding_dim": int(test_embs.shape[1]),
    "encrypt_time_sec": float(encrypt_time),
    "matching_time_sec": float(match_time),
    "eer": float(eer),
    "threshold_at_eer": float(threshold),
    "roc_auc": float(roc_auc),
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "far_at_eer_threshold": float(far),
    "frr_at_eer_threshold": float(frr),
    "tar_at_far_1_percent": float(tar1),
    "actual_far_near_1_percent": float(actual_far1),
    "threshold_near_far_1_percent": float(thr_far1),
    "tar_at_far_0_1_percent": float(tar01),
    "actual_far_near_0_1_percent": float(actual_far01),
    "threshold_near_far_0_1_percent": float(thr_far01),
    "tp": int(TP),
    "tn": int(TN),
    "fp": int(FP),
    "fn": int(FN),
}

with open(os.path.join(SAVE_DIR, "face_ckks_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("\n🔥 FACE CKKS ENCRYPTED MATCHING RESULTS")
print(f"EER              : {eer*100:.2f}%")
print(f"Threshold@EER    : {threshold:.4f}")
print(f"ROC-AUC          : {roc_auc:.4f}")
print(f"Accuracy         : {acc:.4f}")
print(f"Precision        : {prec:.4f}")
print(f"Recall           : {rec:.4f}")
print(f"F1-score         : {f1:.4f}")
print(f"FAR              : {far:.4f}")
print(f"FRR              : {frr:.4f}")
print(f"TAR@FAR=1%       : {tar1:.4f}")
print(f"TAR@FAR=0.1%     : {tar01:.4f}")

# =========================
# 10. SHOW ROC CURVE
# =========================
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Face CKKS ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Face CKKS ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "face_ckks_roc_curve.png"), dpi=200)
plt.close()

# =========================
# 11. SHOW SCORE DISTRIBUTION
# =========================
genuine_scores = scores[labels == 1]
impostor_scores = scores[labels == 0]

plt.figure(figsize=(7, 5))
plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
plt.axvline(threshold, linestyle="--", label=f"Thr@EER={threshold:.4f}")
plt.xlabel("Decrypted CKKS Score")
plt.ylabel("Density")
plt.title("Face CKKS Score Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
plt.axvline(threshold, linestyle="--", label=f"Thr@EER={threshold:.4f}")
plt.xlabel("Decrypted CKKS Score")
plt.ylabel("Density")
plt.title("Face CKKS Score Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "face_ckks_score_distribution.png"), dpi=200)
plt.close()

# =========================
# 12. CONFUSION MATRIX
# =========================
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Face CKKS Confusion Matrix")
plt.colorbar()
ticks = np.arange(2)
plt.xticks(ticks, ["Impostor", "Genuine"])
plt.yticks(ticks, ["Impostor", "Genuine"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Face CKKS Confusion Matrix")
plt.colorbar()
plt.xticks(ticks, ["Impostor", "Genuine"])
plt.yticks(ticks, ["Impostor", "Genuine"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "face_ckks_confusion_matrix.png"), dpi=200)
plt.close()

# =========================
# 13. SAVE ROC POINTS + SUMMARY
# =========================
roc_df = pd.DataFrame({
    "fpr": fpr,
    "tpr": tpr,
    "threshold": thresholds
})
roc_df.to_csv(os.path.join(SAVE_DIR, "face_ckks_roc_points.csv"), index=False)

with open(os.path.join(SAVE_DIR, "face_ckks_summary.txt"), "w") as f:
    f.write("FACE CKKS ENCRYPTED MATCHING SUMMARY\n")
    f.write("===================================\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")

print("\nConfusion Matrix:")
print(cm)

print("\n✅ Saved files:")
for fn in sorted(os.listdir(SAVE_DIR)):
    print(os.path.join(SAVE_DIR, fn))

# ================= NOTEBOOK CELL 40 =================
# ============================================================
# IRIS CKKS ENCRYPTED MATCHING + EVALUATION
# - Uses saved iris test embeddings
# - CKKS encrypted-domain dot-product matching
# - Decrypt scores
# - Evaluate metrics + curves
# ============================================================

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# =========================
# 1. IMPORT TENSEAL
# =========================
import tenseal as ts

# =========================
# 2. PATHS
# =========================
EMB_DIR = "/kaggle/working/iris_balanced_test_outputs"
PAIR_DIR = "/kaggle/working/common_pairs_balanced"
SAVE_DIR = "/kaggle/working/iris_ckks_outputs"

os.makedirs(SAVE_DIR, exist_ok=True)

TEST_EMB_PATH = os.path.join(EMB_DIR, "iris_test_embeddings.npy")
TEST_META_PATH = os.path.join(EMB_DIR, "iris_test_embeddings_meta.csv")
PAIR_CSV = os.path.join(PAIR_DIR, "test_pairs_common_balanced.csv")

# =========================
# 3. HELPERS
# =========================
def clean_sid(x):
    return str(int(float(x))).zfill(3)

def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    return float(eer), float(threshold), fpr, tpr, thresholds

def tar_at_far(labels, scores, target_far=0.01):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmin(np.abs(fpr - target_far))
    return float(tpr[idx]), float(fpr[idx]), float(thresholds[idx])

# =========================
# 4. LOAD EMBEDDINGS + META + PAIRS
# =========================
test_embs = np.load(TEST_EMB_PATH)
test_meta = pd.read_csv(TEST_META_PATH)
pair_df = pd.read_csv(PAIR_CSV)

test_meta["subject"] = test_meta["subject"].apply(clean_sid)
pair_df["subject1"] = pair_df["subject1"].apply(clean_sid)
pair_df["subject2"] = pair_df["subject2"].apply(clean_sid)

print("Embeddings shape:", test_embs.shape)
print("Meta shape      :", test_meta.shape)
print("Pairs shape     :", pair_df.shape)

# =========================
# 5. BUILD EMBEDDING MAP
# =========================
emb_map = {}
for i, row in test_meta.iterrows():
    emb_map[(row["subject"], int(row["img_idx"]))] = test_embs[i].astype(np.float64)

print("Embedding map size:", len(emb_map))

# =========================
# 6. CKKS CONTEXT
# =========================
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

secret_context = context
public_context = context.copy()
public_context.make_context_public()

print("CKKS context ready")

# =========================
# 7. ENCRYPT TEST EMBEDDINGS
# =========================
encrypted_emb_map = {}

start_encrypt = time.time()
for key, vec in emb_map.items():
    encrypted_emb_map[key] = ts.ckks_vector(public_context, vec.tolist())
encrypt_time = time.time() - start_encrypt

print(f"Encrypted {len(encrypted_emb_map)} embeddings in {encrypt_time:.2f} sec")

# =========================
# 8. ENCRYPTED MATCHING
# =========================
scores = []
labels = []
valid_rows = []

start_match = time.time()

for _, row in pair_df.iterrows():
    pair_id = int(row["pair_id"])
    s1 = row["subject1"]
    s2 = row["subject2"]
    idx1 = int(row["idx1"])
    idx2 = int(row["idx2"])
    label = int(row["label"])

    k1 = (s1, idx1)
    k2 = (s2, idx2)

    if k1 not in encrypted_emb_map or k2 not in emb_map:
        continue

    enc_e1 = encrypted_emb_map[k1]
    e2 = emb_map[k2]

    # encrypted dot product
    enc_score = enc_e1.dot(e2.tolist())

    # decrypt scalar
    dec_score = enc_score.decrypt(secret_context.secret_key())

    if isinstance(dec_score, list):
        score = float(dec_score[0])
    else:
        score = float(dec_score)

    scores.append(score)
    labels.append(label)
    valid_rows.append([pair_id, s1, s2, idx1, idx2, label, score])

match_time = time.time() - start_match

score_df = pd.DataFrame(
    valid_rows,
    columns=["pair_id", "subject1", "subject2", "idx1", "idx2", "label", "score"]
)

score_df.to_csv(os.path.join(SAVE_DIR, "iris_test_scores_ckks.csv"), index=False)

scores = np.array(scores, dtype=np.float64)
labels = np.array(labels, dtype=np.int32)

print("Pairs requested :", len(pair_df))
print("Pairs evaluated :", len(score_df))
print(f"Encrypted matching time: {match_time:.2f} sec")

if len(scores) == 0:
    raise ValueError("No valid encrypted iris scores were computed.")

# =========================
# 9. EVALUATION
# =========================
eer, threshold, fpr, tpr, thresholds = compute_eer(scores, labels)
roc_auc = auc(fpr, tpr)

preds = (scores >= threshold).astype(int)

acc = accuracy_score(labels, preds)
prec = precision_score(labels, preds, zero_division=0)
rec = recall_score(labels, preds, zero_division=0)
f1 = f1_score(labels, preds, zero_division=0)

cm = confusion_matrix(labels, preds)
TN, FP, FN, TP = cm.ravel()

far = FP / (FP + TN)
frr = FN / (FN + TP)

tar1, actual_far1, thr_far1 = tar_at_far(labels, scores, target_far=0.01)
tar01, actual_far01, thr_far01 = tar_at_far(labels, scores, target_far=0.001)

metrics = {
    "scheme": "CKKS",
    "library": "TenSEAL",
    "num_pairs_evaluated": int(len(score_df)),
    "embedding_dim": int(test_embs.shape[1]),
    "encrypt_time_sec": float(encrypt_time),
    "matching_time_sec": float(match_time),
    "eer": float(eer),
    "threshold_at_eer": float(threshold),
    "roc_auc": float(roc_auc),
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "far_at_eer_threshold": float(far),
    "frr_at_eer_threshold": float(frr),
    "tar_at_far_1_percent": float(tar1),
    "actual_far_near_1_percent": float(actual_far1),
    "threshold_near_far_1_percent": float(thr_far1),
    "tar_at_far_0_1_percent": float(tar01),
    "actual_far_near_0_1_percent": float(actual_far01),
    "threshold_near_far_0_1_percent": float(thr_far01),
    "tp": int(TP),
    "tn": int(TN),
    "fp": int(FP),
    "fn": int(FN),
}

with open(os.path.join(SAVE_DIR, "iris_ckks_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("\n🔥 IRIS CKKS ENCRYPTED MATCHING RESULTS")
print(f"EER              : {eer*100:.2f}%")
print(f"Threshold@EER    : {threshold:.4f}")
print(f"ROC-AUC          : {roc_auc:.4f}")
print(f"Accuracy         : {acc:.4f}")
print(f"Precision        : {prec:.4f}")
print(f"Recall           : {rec:.4f}")
print(f"F1-score         : {f1:.4f}")
print(f"FAR              : {far:.4f}")
print(f"FRR              : {frr:.4f}")
print(f"TAR@FAR=1%       : {tar1:.4f}")
print(f"TAR@FAR=0.1%     : {tar01:.4f}")

# =========================
# 10. SHOW ROC CURVE
# =========================
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Iris CKKS ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Iris CKKS ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "iris_ckks_roc_curve.png"), dpi=200)
plt.close()

# =========================
# 11. SHOW SCORE DISTRIBUTION
# =========================
genuine_scores = scores[labels == 1]
impostor_scores = scores[labels == 0]

plt.figure(figsize=(7, 5))
plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
plt.axvline(threshold, linestyle="--", label=f"Thr@EER={threshold:.4f}")
plt.xlabel("Decrypted CKKS Score")
plt.ylabel("Density")
plt.title("Iris CKKS Score Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
plt.axvline(threshold, linestyle="--", label=f"Thr@EER={threshold:.4f}")
plt.xlabel("Decrypted CKKS Score")
plt.ylabel("Density")
plt.title("Iris CKKS Score Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "iris_ckks_score_distribution.png"), dpi=200)
plt.close()

# =========================
# 12. CONFUSION MATRIX
# =========================
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Iris CKKS Confusion Matrix")
plt.colorbar()
ticks = np.arange(2)
plt.xticks(ticks, ["Impostor", "Genuine"])
plt.yticks(ticks, ["Impostor", "Genuine"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Iris CKKS Confusion Matrix")
plt.colorbar()
plt.xticks(ticks, ["Impostor", "Genuine"])
plt.yticks(ticks, ["Impostor", "Genuine"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "iris_ckks_confusion_matrix.png"), dpi=200)
plt.close()

# =========================
# 13. SAVE ROC POINTS + SUMMARY
# =========================
roc_df = pd.DataFrame({
    "fpr": fpr,
    "tpr": tpr,
    "threshold": thresholds
})
roc_df.to_csv(os.path.join(SAVE_DIR, "iris_ckks_roc_points.csv"), index=False)

with open(os.path.join(SAVE_DIR, "iris_ckks_summary.txt"), "w") as f:
    f.write("IRIS CKKS ENCRYPTED MATCHING SUMMARY\n")
    f.write("===================================\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")

print("\nConfusion Matrix:")
print(cm)

print("\n✅ Saved files:")
for fn in sorted(os.listdir(SAVE_DIR)):
    print(os.path.join(SAVE_DIR, fn))# ============================================================
# IRIS CKKS ENCRYPTED MATCHING + EVALUATION
# - Uses saved iris test embeddings
# - CKKS encrypted-domain dot-product matching
# - Decrypt scores
# - Evaluate metrics + curves
# ============================================================

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# =========================
# 1. IMPORT TENSEAL
# =========================
import tenseal as ts

# =========================
# 2. PATHS
# =========================
EMB_DIR = "/kaggle/working/iris_balanced_test_outputs"
PAIR_DIR = "/kaggle/working/common_pairs_balanced"
SAVE_DIR = "/kaggle/working/iris_ckks_outputs"

os.makedirs(SAVE_DIR, exist_ok=True)

TEST_EMB_PATH = os.path.join(EMB_DIR, "iris_test_embeddings.npy")
TEST_META_PATH = os.path.join(EMB_DIR, "iris_test_embeddings_meta.csv")
PAIR_CSV = os.path.join(PAIR_DIR, "test_pairs_common_balanced.csv")

# =========================
# 3. HELPERS
# =========================
def clean_sid(x):
    return str(int(float(x))).zfill(3)

def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    return float(eer), float(threshold), fpr, tpr, thresholds

def tar_at_far(labels, scores, target_far=0.01):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmin(np.abs(fpr - target_far))
    return float(tpr[idx]), float(fpr[idx]), float(thresholds[idx])

# =========================
# 4. LOAD EMBEDDINGS + META + PAIRS
# =========================
test_embs = np.load(TEST_EMB_PATH)
test_meta = pd.read_csv(TEST_META_PATH)
pair_df = pd.read_csv(PAIR_CSV)

test_meta["subject"] = test_meta["subject"].apply(clean_sid)
pair_df["subject1"] = pair_df["subject1"].apply(clean_sid)
pair_df["subject2"] = pair_df["subject2"].apply(clean_sid)

print("Embeddings shape:", test_embs.shape)
print("Meta shape      :", test_meta.shape)
print("Pairs shape     :", pair_df.shape)

# =========================
# 5. BUILD EMBEDDING MAP
# =========================
emb_map = {}
for i, row in test_meta.iterrows():
    emb_map[(row["subject"], int(row["img_idx"]))] = test_embs[i].astype(np.float64)

print("Embedding map size:", len(emb_map))

# =========================
# 6. CKKS CONTEXT
# =========================
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

secret_context = context
public_context = context.copy()
public_context.make_context_public()

print("CKKS context ready")

# =========================
# 7. ENCRYPT TEST EMBEDDINGS
# =========================
encrypted_emb_map = {}

start_encrypt = time.time()
for key, vec in emb_map.items():
    encrypted_emb_map[key] = ts.ckks_vector(public_context, vec.tolist())
encrypt_time = time.time() - start_encrypt

print(f"Encrypted {len(encrypted_emb_map)} embeddings in {encrypt_time:.2f} sec")

# =========================
# 8. ENCRYPTED MATCHING
# =========================
scores = []
labels = []
valid_rows = []

start_match = time.time()

for _, row in pair_df.iterrows():
    pair_id = int(row["pair_id"])
    s1 = row["subject1"]
    s2 = row["subject2"]
    idx1 = int(row["idx1"])
    idx2 = int(row["idx2"])
    label = int(row["label"])

    k1 = (s1, idx1)
    k2 = (s2, idx2)

    if k1 not in encrypted_emb_map or k2 not in emb_map:
        continue

    enc_e1 = encrypted_emb_map[k1]
    e2 = emb_map[k2]

    # encrypted dot product
    enc_score = enc_e1.dot(e2.tolist())

    # decrypt scalar
    dec_score = enc_score.decrypt(secret_context.secret_key())

    if isinstance(dec_score, list):
        score = float(dec_score[0])
    else:
        score = float(dec_score)

    scores.append(score)
    labels.append(label)
    valid_rows.append([pair_id, s1, s2, idx1, idx2, label, score])

match_time = time.time() - start_match

score_df = pd.DataFrame(
    valid_rows,
    columns=["pair_id", "subject1", "subject2", "idx1", "idx2", "label", "score"]
)

score_df.to_csv(os.path.join(SAVE_DIR, "iris_test_scores_ckks.csv"), index=False)

scores = np.array(scores, dtype=np.float64)
labels = np.array(labels, dtype=np.int32)

print("Pairs requested :", len(pair_df))
print("Pairs evaluated :", len(score_df))
print(f"Encrypted matching time: {match_time:.2f} sec")

if len(scores) == 0:
    raise ValueError("No valid encrypted iris scores were computed.")

# =========================
# 9. EVALUATION
# =========================
eer, threshold, fpr, tpr, thresholds = compute_eer(scores, labels)
roc_auc = auc(fpr, tpr)

preds = (scores >= threshold).astype(int)

acc = accuracy_score(labels, preds)
prec = precision_score(labels, preds, zero_division=0)
rec = recall_score(labels, preds, zero_division=0)
f1 = f1_score(labels, preds, zero_division=0)

cm = confusion_matrix(labels, preds)
TN, FP, FN, TP = cm.ravel()

far = FP / (FP + TN)
frr = FN / (FN + TP)

tar1, actual_far1, thr_far1 = tar_at_far(labels, scores, target_far=0.01)
tar01, actual_far01, thr_far01 = tar_at_far(labels, scores, target_far=0.001)

metrics = {
    "scheme": "CKKS",
    "library": "TenSEAL",
    "num_pairs_evaluated": int(len(score_df)),
    "embedding_dim": int(test_embs.shape[1]),
    "encrypt_time_sec": float(encrypt_time),
    "matching_time_sec": float(match_time),
    "eer": float(eer),
    "threshold_at_eer": float(threshold),
    "roc_auc": float(roc_auc),
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "far_at_eer_threshold": float(far),
    "frr_at_eer_threshold": float(frr),
    "tar_at_far_1_percent": float(tar1),
    "actual_far_near_1_percent": float(actual_far1),
    "threshold_near_far_1_percent": float(thr_far1),
    "tar_at_far_0_1_percent": float(tar01),
    "actual_far_near_0_1_percent": float(actual_far01),
    "threshold_near_far_0_1_percent": float(thr_far01),
    "tp": int(TP),
    "tn": int(TN),
    "fp": int(FP),
    "fn": int(FN),
}

with open(os.path.join(SAVE_DIR, "iris_ckks_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("\n🔥 IRIS CKKS ENCRYPTED MATCHING RESULTS")
print(f"EER              : {eer*100:.2f}%")
print(f"Threshold@EER    : {threshold:.4f}")
print(f"ROC-AUC          : {roc_auc:.4f}")
print(f"Accuracy         : {acc:.4f}")
print(f"Precision        : {prec:.4f}")
print(f"Recall           : {rec:.4f}")
print(f"F1-score         : {f1:.4f}")
print(f"FAR              : {far:.4f}")
print(f"FRR              : {frr:.4f}")
print(f"TAR@FAR=1%       : {tar1:.4f}")
print(f"TAR@FAR=0.1%     : {tar01:.4f}")

# =========================
# 10. SHOW ROC CURVE
# =========================
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Iris CKKS ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Iris CKKS ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "iris_ckks_roc_curve.png"), dpi=200)
plt.close()

# =========================
# 11. SHOW SCORE DISTRIBUTION
# =========================
genuine_scores = scores[labels == 1]
impostor_scores = scores[labels == 0]

plt.figure(figsize=(7, 5))
plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
plt.axvline(threshold, linestyle="--", label=f"Thr@EER={threshold:.4f}")
plt.xlabel("Decrypted CKKS Score")
plt.ylabel("Density")
plt.title("Iris CKKS Score Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
plt.axvline(threshold, linestyle="--", label=f"Thr@EER={threshold:.4f}")
plt.xlabel("Decrypted CKKS Score")
plt.ylabel("Density")
plt.title("Iris CKKS Score Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "iris_ckks_score_distribution.png"), dpi=200)
plt.close()

# =========================
# 12. CONFUSION MATRIX
# =========================
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Iris CKKS Confusion Matrix")
plt.colorbar()
ticks = np.arange(2)
plt.xticks(ticks, ["Impostor", "Genuine"])
plt.yticks(ticks, ["Impostor", "Genuine"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Iris CKKS Confusion Matrix")
plt.colorbar()
plt.xticks(ticks, ["Impostor", "Genuine"])
plt.yticks(ticks, ["Impostor", "Genuine"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "iris_ckks_confusion_matrix.png"), dpi=200)
plt.close()

# =========================
# 13. SAVE ROC POINTS + SUMMARY
# =========================
roc_df = pd.DataFrame({
    "fpr": fpr,
    "tpr": tpr,
    "threshold": thresholds
})
roc_df.to_csv(os.path.join(SAVE_DIR, "iris_ckks_roc_points.csv"), index=False)

with open(os.path.join(SAVE_DIR, "iris_ckks_summary.txt"), "w") as f:
    f.write("IRIS CKKS ENCRYPTED MATCHING SUMMARY\n")
    f.write("===================================\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")

print("\nConfusion Matrix:")
print(cm)

print("\n✅ Saved files:")
for fn in sorted(os.listdir(SAVE_DIR)):
    print(os.path.join(SAVE_DIR, fn))

# ================= NOTEBOOK CELL 85 =================
# ============================================================
# FINGERPRINT: PLAIN CANCELLABLE + TRUE CKKS ENCRYPTED MATCHING
# ============================================================

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

import tenseal as ts

# ============================================================
# 1. PATHS
# ============================================================

EMB_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/fingerprint_balanced_test_outputs/fingerprint_test_embeddings.npy"
META_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/fingerprint_balanced_test_outputs/fingerprint_test_embeddings_meta.csv"
PAIRS_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/common_pairs_balanced/test_pairs_common_balanced.csv"

OUT_CANC_DIR = "/kaggle/working/fingerprint_cancellable_outputs_trueckks"
OUT_CKKS_DIR = "/kaggle/working/fingerprint_true_ckks_outputs"

os.makedirs(OUT_CANC_DIR, exist_ok=True)
os.makedirs(OUT_CKKS_DIR, exist_ok=True)

# ============================================================
# 2. CONFIG
# ============================================================

KEY = 11
OUT_DIM = 512
BINARY = False

poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
global_scale = 2**40

# ============================================================
# 3. LOAD
# ============================================================

embeddings = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH)
pairs = pd.read_csv(PAIRS_PATH)

print("Embeddings:", embeddings.shape)
print("Meta:", meta.shape)
print("Pairs:", pairs.shape)
print("Meta columns:", list(meta.columns))
print("Pairs columns:", list(pairs.columns))

# ============================================================
# 4. BUILD INDEX MAP
# ============================================================

subject_col = None
for c in meta.columns:
    cl = c.lower()
    if cl in ["subject", "subject_id", "label", "person_id", "id"]:
        subject_col = c
        break
if subject_col is None:
    raise ValueError("Subject column not found in metadata.")

path_col = None
for c in meta.columns:
    cl = c.lower()
    if "path" in cl or "image" in cl or "file" in cl:
        path_col = c
        break
if path_col is None:
    raise ValueError("Image/path column not found in metadata.")

meta = meta.copy()
meta["global_index"] = np.arange(len(meta))
meta = meta.sort_values([subject_col, path_col]).reset_index(drop=True)
meta["local_idx"] = meta.groupby(subject_col).cumcount()

index_map = {
    (int(row[subject_col]), int(row["local_idx"])): int(row["global_index"])
    for _, row in meta.iterrows()
}

print("\nMapping preview:")
print(meta[[subject_col, "local_idx", "global_index"]].head(10))

# ============================================================
# 5. CANCELLABLE TRANSFORM
# ============================================================

def cancellable_transform(emb, key=11, out_dim=512, binary=False):
    rng = np.random.default_rng(seed=key)
    R = rng.standard_normal((emb.shape[1], out_dim))
    proj = emb @ R
    proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)

    if binary:
        return (proj > 0).astype(np.float32)
    return proj.astype(np.float32)

c_embeddings = cancellable_transform(
    embeddings,
    key=KEY,
    out_dim=OUT_DIM,
    binary=BINARY
)

c_emb_path = os.path.join(OUT_CANC_DIR, "fingerprint_test_embeddings_cancellable.npy")
np.save(c_emb_path, c_embeddings)

print("\nSaved cancellable embeddings:")
print(c_emb_path)
print("Shape:", c_embeddings.shape)

# ============================================================
# 6. PLAIN MATCHING
# normalized embeddings => dot product ~= cosine similarity
# ============================================================

plain_scores = []

for _, row in pairs.iterrows():
    s1 = int(row["subject1"])
    s2 = int(row["subject2"])
    i1 = int(row["idx1"])
    i2 = int(row["idx2"])

    g1 = index_map[(s1, i1)]
    g2 = index_map[(s2, i2)]

    score = float(np.dot(c_embeddings[g1], c_embeddings[g2]))
    plain_scores.append(score)

plain_df = pairs.copy()
plain_df["score"] = plain_scores

plain_scores_csv = os.path.join(OUT_CANC_DIR, "fingerprint_test_scores_cancellable_plain.csv")
plain_df.to_csv(plain_scores_csv, index=False)

print("\nSaved plain cancellable scores:")
print(plain_scores_csv)
print(plain_df.head())

# ============================================================
# 7. METRICS HELPER
# ============================================================

def compute_metrics(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(fpr[eer_idx])
    thr = float(thresholds[eer_idx])

    if not np.isfinite(thr):
        thr = 0.0

    auc_val = float(roc_auc_score(y_true, y_score))
    y_pred = (y_score >= thr).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "eer": eer,
        "threshold": thr,
        "auc": auc_val,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "cm": cm
    }

# ============================================================
# 8. PLAIN METRICS
# ============================================================

y_true_plain = plain_df["label"].astype(int).values
y_score_plain = plain_df["score"].astype(float).values

plain_metrics = compute_metrics(y_true_plain, y_score_plain)

print("\n===== PLAIN CANCELLABLE METRICS =====")
for k in ["eer", "threshold", "auc", "accuracy", "precision", "recall", "f1_score"]:
    print(f"{k}: {plain_metrics[k]}")

# ============================================================
# 9. CKKS SETUP
# ============================================================

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=poly_modulus_degree,
    coeff_mod_bit_sizes=coeff_mod_bit_sizes
)
context.generate_galois_keys()
context.global_scale = global_scale

# ============================================================
# 10. ENCRYPT GALLERY TEMPLATES
# ============================================================

print("\nEncrypting gallery templates...")
t0 = time.time()

enc_gallery = []
for i in range(len(c_embeddings)):
    enc_vec = ts.ckks_vector(context, c_embeddings[i].tolist())
    enc_gallery.append(enc_vec)

encrypt_time = time.time() - t0
print(f"Encrypted {len(enc_gallery)} templates in {encrypt_time:.2f} sec")

# ============================================================
# 11. TRUE CKKS MATCHING
# Enc(gallery) dot probe_plain -> decrypt only final scalar
# ============================================================

print("\nScoring pairs with TRUE CKKS encrypted dot product...")
t1 = time.time()

ckks_scores = []
bad_rows = []

for row_id, row in pairs.iterrows():
    s1 = int(row["subject1"])
    s2 = int(row["subject2"])
    i1 = int(row["idx1"])
    i2 = int(row["idx2"])

    k1 = (s1, i1)
    k2 = (s2, i2)

    if k1 not in index_map or k2 not in index_map:
        bad_rows.append((row_id, k1, k2))
        ckks_scores.append(np.nan)
        continue

    probe_idx = index_map[k1]
    gallery_idx = index_map[k2]

    probe_vec = c_embeddings[probe_idx].astype(np.float32)

    enc_score = enc_gallery[gallery_idx].dot(probe_vec.tolist())
    dec_score = enc_score.decrypt()

    if isinstance(dec_score, (list, tuple, np.ndarray)):
        score = float(dec_score[0])
    else:
        score = float(dec_score)

    ckks_scores.append(score)

score_time = time.time() - t1
print(f"Scored {len(pairs)} pairs in {score_time:.2f} sec")

ckks_df = pairs.copy()
ckks_df["score"] = ckks_scores

if bad_rows:
    print("Bad rows:", len(bad_rows))
    print("First few bad rows:", bad_rows[:5])

ckks_df = ckks_df.dropna(subset=["score"]).reset_index(drop=True)

ckks_scores_csv = os.path.join(OUT_CKKS_DIR, "fingerprint_test_scores_true_ckks.csv")
ckks_df.to_csv(ckks_scores_csv, index=False)

print("\nSaved TRUE CKKS scores:")
print(ckks_scores_csv)
print(ckks_df.head())

# ============================================================
# 12. TRUE CKKS METRICS
# ============================================================

y_true_ckks = ckks_df["label"].astype(int).values
y_score_ckks = ckks_df["score"].astype(float).values

ckks_metrics = compute_metrics(y_true_ckks, y_score_ckks)

print("\n===== TRUE CKKS METRICS =====")
for k in ["eer", "threshold", "auc", "accuracy", "precision", "recall", "f1_score"]:
    print(f"{k}: {ckks_metrics[k]}")

# ============================================================
# 13. PLAIN VS CKKS COMPARISON
# ============================================================

compare_df = plain_df[["pair_id", "score"]].rename(columns={"score": "plain_score"})
compare_df = compare_df.merge(
    ckks_df[["pair_id", "score"]].rename(columns={"score": "ckks_score"}),
    on="pair_id",
    how="inner"
)

compare_df["abs_diff"] = np.abs(compare_df["plain_score"] - compare_df["ckks_score"])

mean_abs_diff = float(compare_df["abs_diff"].mean())
max_abs_diff = float(compare_df["abs_diff"].max())

compare_csv = os.path.join(OUT_CKKS_DIR, "fingerprint_plain_vs_trueckks_compare.csv")
compare_df.to_csv(compare_csv, index=False)

print("\n===== PLAIN VS TRUE CKKS COMPARISON =====")
print("Mean abs diff:", mean_abs_diff)
print("Max abs diff :", max_abs_diff)

# ============================================================
# 14. SAVE FINAL METRICS
# ============================================================

cm = ckks_metrics["cm"]
tn, fp, fn, tp = cm.ravel()

final_metrics = {
    "modality": "fingerprint",
    "template_type": "cancellable_true_ckks_dot",
    "key": KEY,
    "out_dim": OUT_DIM,
    "binary": BINARY,
    "plain_eer": plain_metrics["eer"],
    "plain_auc": plain_metrics["auc"],
    "plain_accuracy": plain_metrics["accuracy"],
    "ckks_eer": ckks_metrics["eer"],
    "ckks_auc": ckks_metrics["auc"],
    "ckks_accuracy": ckks_metrics["accuracy"],
    "precision": ckks_metrics["precision"],
    "recall": ckks_metrics["recall"],
    "f1_score": ckks_metrics["f1_score"],
    "threshold": ckks_metrics["threshold"],
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
    "num_pairs": int(len(ckks_df)),
    "poly_modulus_degree": poly_modulus_degree,
    "global_scale": float(global_scale),
    "encryption_time_sec": float(encrypt_time),
    "pair_scoring_time_sec": float(score_time),
    "total_time_sec": float(encrypt_time + score_time),
    "mean_abs_diff_vs_plain": mean_abs_diff,
    "max_abs_diff_vs_plain": max_abs_diff,
}

metrics_json = os.path.join(OUT_CKKS_DIR, "fingerprint_true_ckks_metrics.json")
with open(metrics_json, "w") as f:
    json.dump(final_metrics, f, indent=4)

result_row_csv = os.path.join(OUT_CKKS_DIR, "fingerprint_true_ckks_result_row.csv")
pd.DataFrame([final_metrics]).to_csv(result_row_csv, index=False)

print("\nSaved metrics:")
print(metrics_json)
print(result_row_csv)

# ============================================================
# 15. PLOTS
# ============================================================

fpr = ckks_metrics["fpr"]
tpr = ckks_metrics["tpr"]
thr = ckks_metrics["threshold"]
auc_val = ckks_metrics["auc"]

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Fingerprint TRUE CKKS ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_CKKS_DIR, "fingerprint_true_ckks_roc_curve.png"), dpi=300)
plt.show()

genuine_scores = ckks_df.loc[ckks_df["label"] == 1, "score"].values
impostor_scores = ckks_df.loc[ckks_df["label"] == 0, "score"].values

plt.figure(figsize=(7, 5))
plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
plt.axvline(thr, linestyle="--", label=f"Threshold = {thr:.4f}")
plt.xlabel("Encrypted Dot-Product Score")
plt.ylabel("Density")
plt.title("Fingerprint TRUE CKKS Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_CKKS_DIR, "fingerprint_true_ckks_score_distribution.png"), dpi=300)
plt.show()

plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Fingerprint TRUE CKKS Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["Impostor", "Genuine"])
plt.yticks([0, 1], ["Impostor", "Genuine"])
plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(OUT_CKKS_DIR, "fingerprint_true_ckks_confusion_matrix.png"), dpi=300)
plt.show()

print("\n✅ DONE")
print("Plain scores:", plain_scores_csv)
print("TRUE CKKS scores:", ckks_scores_csv)
print("Compare file:", compare_csv)

# ================= NOTEBOOK CELL 86 =================
# ============================================================
# FACE: PLAIN CANCELLABLE + TRUE CKKS ENCRYPTED MATCHING
# ============================================================

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

import tenseal as ts

# ============================================================
# 1. PATHS
# ============================================================

EMB_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/face_pretrained_embeddings/face_test_embeddings_pretrained.npy"
META_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/face_pretrained_embeddings/face_test_embeddings_meta_pretrained.csv"
PAIRS_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/common_pairs_balanced/test_pairs_common_balanced.csv"

OUT_CANC_DIR = "/kaggle/working/face_cancellable_outputs_trueckks"
OUT_CKKS_DIR = "/kaggle/working/face_true_ckks_outputs"

os.makedirs(OUT_CANC_DIR, exist_ok=True)
os.makedirs(OUT_CKKS_DIR, exist_ok=True)

# ============================================================
# 2. CONFIG
# ============================================================

KEY = 11
OUT_DIM = 512
BINARY = False

poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
global_scale = 2**40

# ============================================================
# 3. LOAD
# ============================================================

embeddings = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH)
pairs = pd.read_csv(PAIRS_PATH)

print("Embeddings:", embeddings.shape)
print("Meta:", meta.shape)
print("Pairs:", pairs.shape)
print("Meta columns:", list(meta.columns))
print("Pairs columns:", list(pairs.columns))

# ============================================================
# 4. BUILD INDEX MAP
# ============================================================

subject_col = None
for c in meta.columns:
    cl = c.lower()
    if cl in ["subject", "subject_id", "label", "person_id", "id"]:
        subject_col = c
        break
if subject_col is None:
    raise ValueError("Subject column not found in metadata.")

path_col = None
for c in meta.columns:
    cl = c.lower()
    if "path" in cl or "image" in cl or "file" in cl:
        path_col = c
        break
if path_col is None:
    raise ValueError("Image/path column not found in metadata.")

meta = meta.copy()
meta["global_index"] = np.arange(len(meta))
meta = meta.sort_values([subject_col, path_col]).reset_index(drop=True)
meta["local_idx"] = meta.groupby(subject_col).cumcount()

index_map = {
    (int(row[subject_col]), int(row["local_idx"])): int(row["global_index"])
    for _, row in meta.iterrows()
}

print("\nMapping preview:")
print(meta[[subject_col, "local_idx", "global_index"]].head(10))

# ============================================================
# 5. CANCELLABLE TRANSFORM
# ============================================================

def cancellable_transform(emb, key=11, out_dim=512, binary=False):
    rng = np.random.default_rng(seed=key)
    R = rng.standard_normal((emb.shape[1], out_dim))
    proj = emb @ R
    proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)

    if binary:
        return (proj > 0).astype(np.float32)
    return proj.astype(np.float32)

c_embeddings = cancellable_transform(
    embeddings,
    key=KEY,
    out_dim=OUT_DIM,
    binary=BINARY
)

c_emb_path = os.path.join(OUT_CANC_DIR, "face_test_embeddings_cancellable.npy")
np.save(c_emb_path, c_embeddings)

print("\nSaved cancellable embeddings:")
print(c_emb_path)
print("Shape:", c_embeddings.shape)

# ============================================================
# 6. PLAIN MATCHING
# ============================================================

plain_scores = []

for _, row in pairs.iterrows():
    s1 = int(row["subject1"])
    s2 = int(row["subject2"])
    i1 = int(row["idx1"])
    i2 = int(row["idx2"])

    g1 = index_map[(s1, i1)]
    g2 = index_map[(s2, i2)]

    score = float(np.dot(c_embeddings[g1], c_embeddings[g2]))
    plain_scores.append(score)

plain_df = pairs.copy()
plain_df["score"] = plain_scores

plain_scores_csv = os.path.join(OUT_CANC_DIR, "face_test_scores_cancellable_plain.csv")
plain_df.to_csv(plain_scores_csv, index=False)

print("\nSaved plain cancellable scores:")
print(plain_scores_csv)
print(plain_df.head())

# ============================================================
# 7. METRICS HELPER
# ============================================================

def compute_metrics(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(fpr[eer_idx])
    thr = float(thresholds[eer_idx])

    if not np.isfinite(thr):
        thr = 0.0

    auc_val = float(roc_auc_score(y_true, y_score))
    y_pred = (y_score >= thr).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "eer": eer,
        "threshold": thr,
        "auc": auc_val,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "cm": cm
    }

# ============================================================
# 8. PLAIN METRICS
# ============================================================

y_true_plain = plain_df["label"].astype(int).values
y_score_plain = plain_df["score"].astype(float).values

plain_metrics = compute_metrics(y_true_plain, y_score_plain)

print("\n===== PLAIN CANCELLABLE METRICS =====")
for k in ["eer", "threshold", "auc", "accuracy", "precision", "recall", "f1_score"]:
    print(f"{k}: {plain_metrics[k]}")

# ============================================================
# 9. CKKS SETUP
# ============================================================

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=poly_modulus_degree,
    coeff_mod_bit_sizes=coeff_mod_bit_sizes
)
context.generate_galois_keys()
context.global_scale = global_scale

# ============================================================
# 10. ENCRYPT GALLERY TEMPLATES
# ============================================================

print("\nEncrypting gallery templates...")
t0 = time.time()

enc_gallery = []
for i in range(len(c_embeddings)):
    enc_vec = ts.ckks_vector(context, c_embeddings[i].tolist())
    enc_gallery.append(enc_vec)

encrypt_time = time.time() - t0
print(f"Encrypted {len(enc_gallery)} templates in {encrypt_time:.2f} sec")

# ============================================================
# 11. TRUE CKKS MATCHING
# ============================================================

print("\nScoring pairs with TRUE CKKS encrypted dot product...")
t1 = time.time()

ckks_scores = []
bad_rows = []

for row_id, row in pairs.iterrows():
    s1 = int(row["subject1"])
    s2 = int(row["subject2"])
    i1 = int(row["idx1"])
    i2 = int(row["idx2"])

    k1 = (s1, i1)
    k2 = (s2, i2)

    if k1 not in index_map or k2 not in index_map:
        bad_rows.append((row_id, k1, k2))
        ckks_scores.append(np.nan)
        continue

    probe_idx = index_map[k1]
    gallery_idx = index_map[k2]

    probe_vec = c_embeddings[probe_idx].astype(np.float32)

    enc_score = enc_gallery[gallery_idx].dot(probe_vec.tolist())
    dec_score = enc_score.decrypt()

    if isinstance(dec_score, (list, tuple, np.ndarray)):
        score = float(dec_score[0])
    else:
        score = float(dec_score)

    ckks_scores.append(score)

score_time = time.time() - t1
print(f"Scored {len(pairs)} pairs in {score_time:.2f} sec")

ckks_df = pairs.copy()
ckks_df["score"] = ckks_scores

if bad_rows:
    print("Bad rows:", len(bad_rows))
    print("First few bad rows:", bad_rows[:5])

ckks_df = ckks_df.dropna(subset=["score"]).reset_index(drop=True)

ckks_scores_csv = os.path.join(OUT_CKKS_DIR, "face_test_scores_true_ckks.csv")
ckks_df.to_csv(ckks_scores_csv, index=False)

print("\nSaved TRUE CKKS scores:")
print(ckks_scores_csv)
print(ckks_df.head())

# ============================================================
# 12. TRUE CKKS METRICS
# ============================================================

y_true_ckks = ckks_df["label"].astype(int).values
y_score_ckks = ckks_df["score"].astype(float).values

ckks_metrics = compute_metrics(y_true_ckks, y_score_ckks)

print("\n===== TRUE CKKS METRICS =====")
for k in ["eer", "threshold", "auc", "accuracy", "precision", "recall", "f1_score"]:
    print(f"{k}: {ckks_metrics[k]}")

# ============================================================
# 13. PLAIN VS CKKS COMPARISON
# ============================================================

compare_df = plain_df[["pair_id", "score"]].rename(columns={"score": "plain_score"})
compare_df = compare_df.merge(
    ckks_df[["pair_id", "score"]].rename(columns={"score": "ckks_score"}),
    on="pair_id",
    how="inner"
)

compare_df["abs_diff"] = np.abs(compare_df["plain_score"] - compare_df["ckks_score"])

mean_abs_diff = float(compare_df["abs_diff"].mean())
max_abs_diff = float(compare_df["abs_diff"].max())

compare_csv = os.path.join(OUT_CKKS_DIR, "face_plain_vs_trueckks_compare.csv")
compare_df.to_csv(compare_csv, index=False)

print("\n===== PLAIN VS TRUE CKKS COMPARISON =====")
print("Mean abs diff:", mean_abs_diff)
print("Max abs diff :", max_abs_diff)

# ============================================================
# 14. SAVE FINAL METRICS
# ============================================================

cm = ckks_metrics["cm"]
tn, fp, fn, tp = cm.ravel()

final_metrics = {
    "modality": "face",
    "template_type": "cancellable_true_ckks_dot",
    "key": KEY,
    "out_dim": OUT_DIM,
    "binary": BINARY,
    "plain_eer": plain_metrics["eer"],
    "plain_auc": plain_metrics["auc"],
    "plain_accuracy": plain_metrics["accuracy"],
    "ckks_eer": ckks_metrics["eer"],
    "ckks_auc": ckks_metrics["auc"],
    "ckks_accuracy": ckks_metrics["accuracy"],
    "precision": ckks_metrics["precision"],
    "recall": ckks_metrics["recall"],
    "f1_score": ckks_metrics["f1_score"],
    "threshold": ckks_metrics["threshold"],
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
    "num_pairs": int(len(ckks_df)),
    "poly_modulus_degree": poly_modulus_degree,
    "global_scale": float(global_scale),
    "encryption_time_sec": float(encrypt_time),
    "pair_scoring_time_sec": float(score_time),
    "total_time_sec": float(encrypt_time + score_time),
    "mean_abs_diff_vs_plain": mean_abs_diff,
    "max_abs_diff_vs_plain": max_abs_diff,
}

metrics_json = os.path.join(OUT_CKKS_DIR, "face_true_ckks_metrics.json")
with open(metrics_json, "w") as f:
    json.dump(final_metrics, f, indent=4)

result_row_csv = os.path.join(OUT_CKKS_DIR, "face_true_ckks_result_row.csv")
pd.DataFrame([final_metrics]).to_csv(result_row_csv, index=False)

print("\nSaved metrics:")
print(metrics_json)
print(result_row_csv)

# ============================================================
# 15. PLOTS
# ============================================================

fpr = ckks_metrics["fpr"]
tpr = ckks_metrics["tpr"]
thr = ckks_metrics["threshold"]
auc_val = ckks_metrics["auc"]

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Face TRUE CKKS ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_CKKS_DIR, "face_true_ckks_roc_curve.png"), dpi=300)
plt.show()

genuine_scores = ckks_df.loc[ckks_df["label"] == 1, "score"].values
impostor_scores = ckks_df.loc[ckks_df["label"] == 0, "score"].values

plt.figure(figsize=(7, 5))
plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
plt.axvline(thr, linestyle="--", label=f"Threshold = {thr:.4f}")
plt.xlabel("Encrypted Dot-Product Score")
plt.ylabel("Density")
plt.title("Face TRUE CKKS Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_CKKS_DIR, "face_true_ckks_score_distribution.png"), dpi=300)
plt.show()

plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Face TRUE CKKS Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["Impostor", "Genuine"])
plt.yticks([0, 1], ["Impostor", "Genuine"])
plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(OUT_CKKS_DIR, "face_true_ckks_confusion_matrix.png"), dpi=300)
plt.show()

print("\n✅ DONE")
print("Plain scores:", plain_scores_csv)
print("TRUE CKKS scores:", ckks_scores_csv)
print("Compare file:", compare_csv)

# ================= NOTEBOOK CELL 87 =================
# ============================================================
# IRIS: PLAIN CANCELLABLE + TRUE CKKS ENCRYPTED MATCHING
# ============================================================

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

import tenseal as ts

# ============================================================
# 1. PATHS
# ============================================================

EMB_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/iris_balanced_test_outputs/iris_test_embeddings.npy"
META_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/iris_balanced_test_outputs/iris_test_embeddings_meta.csv"
PAIRS_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/common_pairs_balanced/test_pairs_common_balanced.csv"

OUT_CANC_DIR = "/kaggle/working/iris_cancellable_outputs_trueckks"
OUT_CKKS_DIR = "/kaggle/working/iris_true_ckks_outputs"

os.makedirs(OUT_CANC_DIR, exist_ok=True)
os.makedirs(OUT_CKKS_DIR, exist_ok=True)

# ============================================================
# 2. CONFIG
# ============================================================

KEY = 11
OUT_DIM = 512
BINARY = False

poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
global_scale = 2**40

# ============================================================
# 3. LOAD
# ============================================================

embeddings = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH)
pairs = pd.read_csv(PAIRS_PATH)

print("Embeddings:", embeddings.shape)
print("Meta:", meta.shape)
print("Pairs:", pairs.shape)
print("Meta columns:", list(meta.columns))
print("Pairs columns:", list(pairs.columns))

# ============================================================
# 4. BUILD INDEX MAP
# ============================================================

subject_col = None
for c in meta.columns:
    cl = c.lower()
    if cl in ["subject", "subject_id", "label", "person_id", "id"]:
        subject_col = c
        break
if subject_col is None:
    raise ValueError("Subject column not found in metadata.")

path_col = None
for c in meta.columns:
    cl = c.lower()
    if "path" in cl or "image" in cl or "file" in cl:
        path_col = c
        break
if path_col is None:
    raise ValueError("Image/path column not found in metadata.")

meta = meta.copy()
meta["global_index"] = np.arange(len(meta))
meta = meta.sort_values([subject_col, path_col]).reset_index(drop=True)
meta["local_idx"] = meta.groupby(subject_col).cumcount()

index_map = {
    (int(row[subject_col]), int(row["local_idx"])): int(row["global_index"])
    for _, row in meta.iterrows()
}

print("\nMapping preview:")
print(meta[[subject_col, "local_idx", "global_index"]].head(10))

# ============================================================
# 5. CANCELLABLE TRANSFORM
# ============================================================

def cancellable_transform(emb, key=11, out_dim=512, binary=False):
    rng = np.random.default_rng(seed=key)
    R = rng.standard_normal((emb.shape[1], out_dim))
    proj = emb @ R
    proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)

    if binary:
        return (proj > 0).astype(np.float32)
    return proj.astype(np.float32)

c_embeddings = cancellable_transform(
    embeddings,
    key=KEY,
    out_dim=OUT_DIM,
    binary=BINARY
)

c_emb_path = os.path.join(OUT_CANC_DIR, "iris_test_embeddings_cancellable.npy")
np.save(c_emb_path, c_embeddings)

print("\nSaved cancellable embeddings:")
print(c_emb_path)
print("Shape:", c_embeddings.shape)

# ============================================================
# 6. PLAIN MATCHING
# ============================================================

plain_scores = []

for _, row in pairs.iterrows():
    s1 = int(row["subject1"])
    s2 = int(row["subject2"])
    i1 = int(row["idx1"])
    i2 = int(row["idx2"])

    g1 = index_map[(s1, i1)]
    g2 = index_map[(s2, i2)]

    score = float(np.dot(c_embeddings[g1], c_embeddings[g2]))
    plain_scores.append(score)

plain_df = pairs.copy()
plain_df["score"] = plain_scores

plain_scores_csv = os.path.join(OUT_CANC_DIR, "iris_test_scores_cancellable_plain.csv")
plain_df.to_csv(plain_scores_csv, index=False)

print("\nSaved plain cancellable scores:")
print(plain_scores_csv)
print(plain_df.head())

# ============================================================
# 7. METRICS HELPER
# ============================================================

def compute_metrics(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(fpr[eer_idx])
    thr = float(thresholds[eer_idx])

    if not np.isfinite(thr):
        thr = 0.0

    auc_val = float(roc_auc_score(y_true, y_score))
    y_pred = (y_score >= thr).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "eer": eer,
        "threshold": thr,
        "auc": auc_val,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "cm": cm
    }

# ============================================================
# 8. PLAIN METRICS
# ============================================================

y_true_plain = plain_df["label"].astype(int).values
y_score_plain = plain_df["score"].astype(float).values

plain_metrics = compute_metrics(y_true_plain, y_score_plain)

print("\n===== PLAIN CANCELLABLE METRICS =====")
for k in ["eer", "threshold", "auc", "accuracy", "precision", "recall", "f1_score"]:
    print(f"{k}: {plain_metrics[k]}")

# ============================================================
# 9. CKKS SETUP
# ============================================================

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=poly_modulus_degree,
    coeff_mod_bit_sizes=coeff_mod_bit_sizes
)
context.generate_galois_keys()
context.global_scale = global_scale

# ============================================================
# 10. ENCRYPT GALLERY TEMPLATES
# ============================================================

print("\nEncrypting gallery templates...")
t0 = time.time()

enc_gallery = []
for i in range(len(c_embeddings)):
    enc_vec = ts.ckks_vector(context, c_embeddings[i].tolist())
    enc_gallery.append(enc_vec)

encrypt_time = time.time() - t0
print(f"Encrypted {len(enc_gallery)} templates in {encrypt_time:.2f} sec")

# ============================================================
# 11. TRUE CKKS MATCHING
# ============================================================

print("\nScoring pairs with TRUE CKKS encrypted dot product...")
t1 = time.time()

ckks_scores = []
bad_rows = []

for row_id, row in pairs.iterrows():
    s1 = int(row["subject1"])
    s2 = int(row["subject2"])
    i1 = int(row["idx1"])
    i2 = int(row["idx2"])

    k1 = (s1, i1)
    k2 = (s2, i2)

    if k1 not in index_map or k2 not in index_map:
        bad_rows.append((row_id, k1, k2))
        ckks_scores.append(np.nan)
        continue

    probe_idx = index_map[k1]
    gallery_idx = index_map[k2]

    probe_vec = c_embeddings[probe_idx].astype(np.float32)

    enc_score = enc_gallery[gallery_idx].dot(probe_vec.tolist())
    dec_score = enc_score.decrypt()

    if isinstance(dec_score, (list, tuple, np.ndarray)):
        score = float(dec_score[0])
    else:
        score = float(dec_score)

    ckks_scores.append(score)

score_time = time.time() - t1
print(f"Scored {len(pairs)} pairs in {score_time:.2f} sec")

ckks_df = pairs.copy()
ckks_df["score"] = ckks_scores

if bad_rows:
    print("Bad rows:", len(bad_rows))
    print("First few bad rows:", bad_rows[:5])

ckks_df = ckks_df.dropna(subset=["score"]).reset_index(drop=True)

ckks_scores_csv = os.path.join(OUT_CKKS_DIR, "iris_test_scores_true_ckks.csv")
ckks_df.to_csv(ckks_scores_csv, index=False)

print("\nSaved TRUE CKKS scores:")
print(ckks_scores_csv)
print(ckks_df.head())

# ============================================================
# 12. TRUE CKKS METRICS
# ============================================================

y_true_ckks = ckks_df["label"].astype(int).values
y_score_ckks = ckks_df["score"].astype(float).values

ckks_metrics = compute_metrics(y_true_ckks, y_score_ckks)

print("\n===== TRUE CKKS METRICS =====")
for k in ["eer", "threshold", "auc", "accuracy", "precision", "recall", "f1_score"]:
    print(f"{k}: {ckks_metrics[k]}")

# ============================================================
# 13. PLAIN VS CKKS COMPARISON
# ============================================================

compare_df = plain_df[["pair_id", "score"]].rename(columns={"score": "plain_score"})
compare_df = compare_df.merge(
    ckks_df[["pair_id", "score"]].rename(columns={"score": "ckks_score"}),
    on="pair_id",
    how="inner"
)

compare_df["abs_diff"] = np.abs(compare_df["plain_score"] - compare_df["ckks_score"])

mean_abs_diff = float(compare_df["abs_diff"].mean())
max_abs_diff = float(compare_df["abs_diff"].max())

compare_csv = os.path.join(OUT_CKKS_DIR, "iris_plain_vs_trueckks_compare.csv")
compare_df.to_csv(compare_csv, index=False)

print("\n===== PLAIN VS TRUE CKKS COMPARISON =====")
print("Mean abs diff:", mean_abs_diff)
print("Max abs diff :", max_abs_diff)

# ============================================================
# 14. SAVE FINAL METRICS
# ============================================================

cm = ckks_metrics["cm"]
tn, fp, fn, tp = cm.ravel()

final_metrics = {
    "modality": "iris",
    "template_type": "cancellable_true_ckks_dot",
    "key": KEY,
    "out_dim": OUT_DIM,
    "binary": BINARY,
    "plain_eer": plain_metrics["eer"],
    "plain_auc": plain_metrics["auc"],
    "plain_accuracy": plain_metrics["accuracy"],
    "ckks_eer": ckks_metrics["eer"],
    "ckks_auc": ckks_metrics["auc"],
    "ckks_accuracy": ckks_metrics["accuracy"],
    "precision": ckks_metrics["precision"],
    "recall": ckks_metrics["recall"],
    "f1_score": ckks_metrics["f1_score"],
    "threshold": ckks_metrics["threshold"],
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
    "num_pairs": int(len(ckks_df)),
    "poly_modulus_degree": poly_modulus_degree,
    "global_scale": float(global_scale),
    "encryption_time_sec": float(encrypt_time),
    "pair_scoring_time_sec": float(score_time),
    "total_time_sec": float(encrypt_time + score_time),
    "mean_abs_diff_vs_plain": mean_abs_diff,
    "max_abs_diff_vs_plain": max_abs_diff,
}

metrics_json = os.path.join(OUT_CKKS_DIR, "iris_true_ckks_metrics.json")
with open(metrics_json, "w") as f:
    json.dump(final_metrics, f, indent=4)

result_row_csv = os.path.join(OUT_CKKS_DIR, "iris_true_ckks_result_row.csv")
pd.DataFrame([final_metrics]).to_csv(result_row_csv, index=False)

print("\nSaved metrics:")
print(metrics_json)
print(result_row_csv)

# ============================================================
# 15. PLOTS
# ============================================================

fpr = ckks_metrics["fpr"]
tpr = ckks_metrics["tpr"]
thr = ckks_metrics["threshold"]
auc_val = ckks_metrics["auc"]

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Iris TRUE CKKS ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_CKKS_DIR, "iris_true_ckks_roc_curve.png"), dpi=300)
plt.show()

genuine_scores = ckks_df.loc[ckks_df["label"] == 1, "score"].values
impostor_scores = ckks_df.loc[ckks_df["label"] == 0, "score"].values

plt.figure(figsize=(7, 5))
plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
plt.axvline(thr, linestyle="--", label=f"Threshold = {thr:.4f}")
plt.xlabel("Encrypted Dot-Product Score")
plt.ylabel("Density")
plt.title("Iris TRUE CKKS Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_CKKS_DIR, "iris_true_ckks_score_distribution.png"), dpi=300)
plt.show()

plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Iris TRUE CKKS Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["Impostor", "Genuine"])
plt.yticks([0, 1], ["Impostor", "Genuine"])
plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(OUT_CKKS_DIR, "iris_true_ckks_confusion_matrix.png"), dpi=300)
plt.show()

print("\n✅ DONE")
print("Plain scores:", plain_scores_csv)
print("TRUE CKKS scores:", ckks_scores_csv)
print("Compare file:", compare_csv)
