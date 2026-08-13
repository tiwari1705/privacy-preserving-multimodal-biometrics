"""Extracted from notebook3e7611700a(1).ipynb.
Cells preserved in original execution order.
Kaggle paths are intentionally preserved; replace them for local execution.
"""

# ================= NOTEBOOK CELL 43 =================
# ============================================================
# MULTIMODAL SCORE-LEVEL FUSION
# Plain domain + Encrypted-domain score fusion
# Pairs:
#   1. Face + Fingerprint
#   2. Face + Iris
#   3. Fingerprint + Iris
#   4. Face + Fingerprint + Iris
# ============================================================

import os
import json
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

# ============================================================
# 1. PATHS
# ============================================================
SAVE_DIR = "/kaggle/working/fusion_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- PLAIN ----------
FP_PLAIN = "/kaggle/working/fingerprint_balanced_test_outputs/fingerprint_test_scores_balanced.csv"
FACE_PLAIN = "/kaggle/working/face_pretrained_test_outputs/face_test_scores_pretrained_balanced.csv"
IRIS_PLAIN = "/kaggle/working/iris_balanced_test_outputs/iris_test_scores_balanced.csv"

# ---------- ENCRYPTED ----------
FP_CKKS = "/kaggle/working/fingerprint_ckks_outputs/fingerprint_test_scores_ckks.csv"
FACE_CKKS = "/kaggle/working/face_ckks_outputs/face_test_scores_ckks.csv"
IRIS_CKKS = "/kaggle/working/iris_ckks_outputs/iris_test_scores_ckks.csv"

# ============================================================
# 2. HELPERS
# ============================================================
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

def minmax_norm(s):
    s = np.asarray(s, dtype=np.float64)
    mn, mx = s.min(), s.max()
    if abs(mx - mn) < 1e-12:
        return np.zeros_like(s)
    return (s - mn) / (mx - mn)

def zscore_norm(s):
    s = np.asarray(s, dtype=np.float64)
    mu, std = s.mean(), s.std()
    if std < 1e-12:
        return np.zeros_like(s)
    return (s - mu) / std

def evaluate_scores(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

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

    extra = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "cm": cm,
        "preds": preds
    }

    return metrics, extra

def show_and_save_curves(labels, scores, metrics, extra, prefix):
    fpr = extra["fpr"]
    tpr = extra["tpr"]
    cm = extra["cm"]
    thr = metrics["threshold_at_eer"]

    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    # ROC
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{prefix}_roc_curve.png"), dpi=200)
    plt.close()

    # score distribution
    plt.figure(figsize=(7,5))
    plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
    plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
    plt.axvline(thr, linestyle="--", label=f"Thr@EER={thr:.4f}")
    plt.xlabel("Fusion Score")
    plt.ylabel("Density")
    plt.title(f"{prefix} Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,5))
    plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
    plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
    plt.axvline(thr, linestyle="--", label=f"Thr@EER={thr:.4f}")
    plt.xlabel("Fusion Score")
    plt.ylabel("Density")
    plt.title(f"{prefix} Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{prefix}_score_distribution.png"), dpi=200)
    plt.close()

    # confusion matrix
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{prefix} Confusion Matrix")
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

    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{prefix} Confusion Matrix")
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
    plt.savefig(os.path.join(SAVE_DIR, f"{prefix}_confusion_matrix.png"), dpi=200)
    plt.close()

    # roc points
    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": extra["thresholds"]
    })
    roc_df.to_csv(os.path.join(SAVE_DIR, f"{prefix}_roc_points.csv"), index=False)

def load_score_csv(path, modality_name):
    df = pd.read_csv(path)
    keep_cols = ["pair_id", "label", "score"]
    df = df[keep_cols].copy()
    df = df.rename(columns={"score": modality_name})
    return df

def fuse_and_evaluate(base_df, score_cols, weights, prefix):
    work = base_df.copy()

    # normalize each score column with min-max
    for c in score_cols:
        work[c + "_norm"] = minmax_norm(work[c].values)

    fused = np.zeros(len(work), dtype=np.float64)
    for c, w in zip(score_cols, weights):
        fused += w * work[c + "_norm"].values

    labels = work["label"].values
    work["fusion_score"] = fused

    metrics, extra = evaluate_scores(labels, fused)

    work.to_csv(os.path.join(SAVE_DIR, f"{prefix}_scores.csv"), index=False)

    with open(os.path.join(SAVE_DIR, f"{prefix}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n🔥 {prefix}")
    print(f"EER              : {metrics['eer']*100:.2f}%")
    print(f"Threshold@EER    : {metrics['threshold_at_eer']:.4f}")
    print(f"ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Precision        : {metrics['precision']:.4f}")
    print(f"Recall           : {metrics['recall']:.4f}")
    print(f"F1-score         : {metrics['f1_score']:.4f}")
    print(f"FAR              : {metrics['far_at_eer_threshold']:.4f}")
    print(f"FRR              : {metrics['frr_at_eer_threshold']:.4f}")
    print(f"TAR@FAR=1%       : {metrics['tar_at_far_1_percent']:.4f}")
    print(f"TAR@FAR=0.1%     : {metrics['tar_at_far_0_1_percent']:.4f}")

    show_and_save_curves(labels, fused, metrics, extra, prefix)

    return metrics

# ============================================================
# 3. LOAD PLAIN SCORES
# ============================================================
fp_plain = load_score_csv(FP_PLAIN, "fingerprint")
face_plain = load_score_csv(FACE_PLAIN, "face")
iris_plain = load_score_csv(IRIS_PLAIN, "iris")

plain = fp_plain.merge(face_plain, on=["pair_id", "label"])
plain = plain.merge(iris_plain, on=["pair_id", "label"])

print("Plain merged shape:", plain.shape)

# ============================================================
# 4. LOAD ENCRYPTED SCORES
# ============================================================
fp_ckks = load_score_csv(FP_CKKS, "fingerprint")
face_ckks = load_score_csv(FACE_CKKS, "face")
iris_ckks = load_score_csv(IRIS_CKKS, "iris")

enc = fp_ckks.merge(face_ckks, on=["pair_id", "label"])
enc = enc.merge(iris_ckks, on=["pair_id", "label"])

print("Encrypted merged shape:", enc.shape)

# ============================================================
# 5. PLAIN DOMAIN FUSION
# Weights can be tuned. Strong initial choice:
# face = 0.4, iris = 0.4, fingerprint = 0.2
# ============================================================
plain_metrics = {}

plain_metrics["plain_face_fingerprint"] = fuse_and_evaluate(
    plain[["pair_id", "label", "face", "fingerprint"]].copy(),
    score_cols=["face", "fingerprint"],
    weights=[0.7, 0.3],
    prefix="plain_face_fingerprint"
)

plain_metrics["plain_face_iris"] = fuse_and_evaluate(
    plain[["pair_id", "label", "face", "iris"]].copy(),
    score_cols=["face", "iris"],
    weights=[0.5, 0.5],
    prefix="plain_face_iris"
)

plain_metrics["plain_fingerprint_iris"] = fuse_and_evaluate(
    plain[["pair_id", "label", "fingerprint", "iris"]].copy(),
    score_cols=["fingerprint", "iris"],
    weights=[0.3, 0.7],
    prefix="plain_fingerprint_iris"
)

plain_metrics["plain_all_three"] = fuse_and_evaluate(
    plain[["pair_id", "label", "face", "fingerprint", "iris"]].copy(),
    score_cols=["face", "fingerprint", "iris"],
    weights=[0.4, 0.2, 0.4],
    prefix="plain_all_three"
)

# ============================================================
# 6. ENCRYPTED-DOMAIN SCORE FUSION
# Here encrypted matching already happened; saved scores are fused now.
# ============================================================
enc_metrics = {}

enc_metrics["enc_face_fingerprint"] = fuse_and_evaluate(
    enc[["pair_id", "label", "face", "fingerprint"]].copy(),
    score_cols=["face", "fingerprint"],
    weights=[0.7, 0.3],
    prefix="enc_face_fingerprint"
)

enc_metrics["enc_face_iris"] = fuse_and_evaluate(
    enc[["pair_id", "label", "face", "iris"]].copy(),
    score_cols=["face", "iris"],
    weights=[0.5, 0.5],
    prefix="enc_face_iris"
)

enc_metrics["enc_fingerprint_iris"] = fuse_and_evaluate(
    enc[["pair_id", "label", "fingerprint", "iris"]].copy(),
    score_cols=["fingerprint", "iris"],
    weights=[0.3, 0.7],
    prefix="enc_fingerprint_iris"
)

enc_metrics["enc_all_three"] = fuse_and_evaluate(
    enc[["pair_id", "label", "face", "fingerprint", "iris"]].copy(),
    score_cols=["face", "fingerprint", "iris"],
    weights=[0.4, 0.2, 0.4],
    prefix="enc_all_three"
)

# ============================================================
# 7. SUMMARY TABLE
# ============================================================
summary_rows = []

for name, met in plain_metrics.items():
    summary_rows.append([
        name,
        met["eer"],
        met["roc_auc"],
        met["accuracy"],
        met["tar_at_far_1_percent"],
        met["tar_at_far_0_1_percent"]
    ])

for name, met in enc_metrics.items():
    summary_rows.append([
        name,
        met["eer"],
        met["roc_auc"],
        met["accuracy"],
        met["tar_at_far_1_percent"],
        met["tar_at_far_0_1_percent"]
    ])

summary_df = pd.DataFrame(
    summary_rows,
    columns=[
        "fusion_name",
        "eer",
        "roc_auc",
        "accuracy",
        "tar_at_far_1_percent",
        "tar_at_far_0_1_percent"
    ]
)

summary_df.to_csv(os.path.join(SAVE_DIR, "fusion_summary.csv"), index=False)

print("\n================ FUSION SUMMARY ================")
print(summary_df.sort_values("eer"))
print("\n✅ Saved fusion summary to:", os.path.join(SAVE_DIR, "fusion_summary.csv"))
print("✅ DONE")

# ================= NOTEBOOK CELL 47 =================
# ============================================================
# STEP 1: FUSION WEIGHT OPTIMIZATION ON VALIDATION SCORES
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

SAVE_DIR = "/kaggle/working/fusion_weight_search"
os.makedirs(SAVE_DIR, exist_ok=True)

FP_VAL = "/kaggle/working/fingerprint_training_balanced/best_val_scores.csv"
FACE_VAL = "/kaggle/working/face_training_balanced/best_val_scores.csv"
IRIS_VAL = "/kaggle/working/iris_training_balanced/best_val_scores.csv"

def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = thresholds[idx]
    return float(eer), float(thr)

def minmax_norm(x):
    x = np.asarray(x, dtype=np.float64)
    mn, mx = x.min(), x.max()
    if abs(mx - mn) < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def load_score_csv(path, name):
    df = pd.read_csv(path)
    df = df[["pair_id", "label", "score"]].copy()
    df = df.rename(columns={"score": name})
    return df

fp = load_score_csv(FP_VAL, "fingerprint")
face = load_score_csv(FACE_VAL, "face")
iris = load_score_csv(IRIS_VAL, "iris")

df = fp.merge(face, on=["pair_id", "label"])
df = df.merge(iris, on=["pair_id", "label"])

print("Merged val shape:", df.shape)

df["fingerprint_n"] = minmax_norm(df["fingerprint"].values)
df["face_n"] = minmax_norm(df["face"].values)
df["iris_n"] = minmax_norm(df["iris"].values)

labels = df["label"].values

results = []
grid = np.arange(0.0, 1.01, 0.05)

for w_fp in grid:
    for w_face in grid:
        for w_iris in grid:
            s = w_fp + w_face + w_iris
            if abs(s - 1.0) > 1e-9:
                continue

            fused = (
                w_fp * df["fingerprint_n"].values +
                w_face * df["face_n"].values +
                w_iris * df["iris_n"].values
            )

            eer, thr = compute_eer(fused, labels)
            results.append([w_fp, w_face, w_iris, eer, thr])

results_df = pd.DataFrame(
    results,
    columns=["w_fingerprint", "w_face", "w_iris", "eer", "threshold"]
)

results_df = results_df.sort_values("eer").reset_index(drop=True)
results_df.to_csv(os.path.join(SAVE_DIR, "fusion_weight_search_results.csv"), index=False)

best = results_df.iloc[0].to_dict()

with open(os.path.join(SAVE_DIR, "best_fusion_weights.json"), "w") as f:
    json.dump(best, f, indent=4)

print("\nTop 10 weight combinations:")
print(results_df.head(10))

print("\n✅ Best weights found:")
print(best)

# ================= NOTEBOOK CELL 48 =================
# ============================================================
# FINAL OPTIMIZED MULTIMODAL FUSION
# - Applies validation-optimized weights to TEST scores
# - Runs both PLAIN and ENCRYPTED score fusion
# - Saves metrics, scores, curves, confusion matrices
# ============================================================

import os
import json
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

# ============================================================
# 1. PATHS
# ============================================================
SAVE_DIR = "/kaggle/working/final_optimized_fusion_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- PLAIN TEST SCORES ----------
FP_PLAIN = "/kaggle/working/fingerprint_balanced_test_outputs/fingerprint_test_scores_balanced.csv"
FACE_PLAIN = "/kaggle/working/face_pretrained_test_outputs/face_test_scores_pretrained_balanced.csv"
IRIS_PLAIN = "/kaggle/working/iris_balanced_test_outputs/iris_test_scores_balanced.csv"

# ---------- ENCRYPTED TEST SCORES ----------
FP_CKKS = "/kaggle/working/fingerprint_ckks_outputs/fingerprint_test_scores_ckks.csv"
FACE_CKKS = "/kaggle/working/face_ckks_outputs/face_test_scores_ckks.csv"
IRIS_CKKS = "/kaggle/working/iris_ckks_outputs/iris_test_scores_ckks.csv"

# ============================================================
# 2. OPTIMIZED WEIGHTS FROM VALIDATION SEARCH
# ============================================================
W_FP = 0.30
W_FACE = 0.35
W_IRIS = 0.35

# ============================================================
# 3. HELPERS
# ============================================================
def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = thresholds[idx]
    return float(eer), float(thr), fpr, tpr, thresholds

def tar_at_far(labels, scores, target_far=0.01):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmin(np.abs(fpr - target_far))
    return float(tpr[idx]), float(fpr[idx]), float(thresholds[idx])

def minmax_norm(x):
    x = np.asarray(x, dtype=np.float64)
    mn, mx = x.min(), x.max()
    if abs(mx - mn) < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def load_score_csv(path, name):
    df = pd.read_csv(path)
    df = df[["pair_id", "label", "score"]].copy()
    df = df.rename(columns={"score": name})
    return df

def evaluate_scores(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

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

    extra = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "cm": cm
    }
    return metrics, extra

def show_and_save_curves(labels, scores, metrics, extra, prefix):
    fpr = extra["fpr"]
    tpr = extra["tpr"]
    cm = extra["cm"]
    thr = metrics["threshold_at_eer"]

    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    # ROC
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{prefix}_roc_curve.png"), dpi=200)
    plt.close()

    # score distribution
    plt.figure(figsize=(7,5))
    plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
    plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
    plt.axvline(thr, linestyle="--", label=f"Thr@EER={thr:.4f}")
    plt.xlabel("Fusion Score")
    plt.ylabel("Density")
    plt.title(f"{prefix} Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,5))
    plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
    plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
    plt.axvline(thr, linestyle="--", label=f"Thr@EER={thr:.4f}")
    plt.xlabel("Fusion Score")
    plt.ylabel("Density")
    plt.title(f"{prefix} Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{prefix}_score_distribution.png"), dpi=200)
    plt.close()

    # confusion matrix
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{prefix} Confusion Matrix")
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

    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{prefix} Confusion Matrix")
    plt.colorbar()
    plt.xticks(ticks, ["Impostor", "Genuine"])
    plt.yticks(ticks, ["Impostor", "Genuine"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{prefix}_confusion_matrix.png"), dpi=200)
    plt.close()

    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": extra["thresholds"]
    })
    roc_df.to_csv(os.path.join(SAVE_DIR, f"{prefix}_roc_points.csv"), index=False)

def fuse_three(df, prefix):
    work = df.copy()

    work["fingerprint_n"] = minmax_norm(work["fingerprint"].values)
    work["face_n"] = minmax_norm(work["face"].values)
    work["iris_n"] = minmax_norm(work["iris"].values)

    work["fusion_score"] = (
        W_FP   * work["fingerprint_n"].values +
        W_FACE * work["face_n"].values +
        W_IRIS * work["iris_n"].values
    )

    labels = work["label"].values
    scores = work["fusion_score"].values

    metrics, extra = evaluate_scores(labels, scores)

    work.to_csv(os.path.join(SAVE_DIR, f"{prefix}_scores.csv"), index=False)

    with open(os.path.join(SAVE_DIR, f"{prefix}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n🔥 {prefix}")
    print(f"Weights          : FP={W_FP}, FACE={W_FACE}, IRIS={W_IRIS}")
    print(f"EER              : {metrics['eer']*100:.2f}%")
    print(f"Threshold@EER    : {metrics['threshold_at_eer']:.4f}")
    print(f"ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Precision        : {metrics['precision']:.4f}")
    print(f"Recall           : {metrics['recall']:.4f}")
    print(f"F1-score         : {metrics['f1_score']:.4f}")
    print(f"FAR              : {metrics['far_at_eer_threshold']:.4f}")
    print(f"FRR              : {metrics['frr_at_eer_threshold']:.4f}")
    print(f"TAR@FAR=1%       : {metrics['tar_at_far_1_percent']:.4f}")
    print(f"TAR@FAR=0.1%     : {metrics['tar_at_far_0_1_percent']:.4f}")

    show_and_save_curves(labels, scores, metrics, extra, prefix)
    return metrics

# ============================================================
# 4. LOAD AND MERGE PLAIN SCORES
# ============================================================
fp_plain = load_score_csv(FP_PLAIN, "fingerprint")
face_plain = load_score_csv(FACE_PLAIN, "face")
iris_plain = load_score_csv(IRIS_PLAIN, "iris")

plain = fp_plain.merge(face_plain, on=["pair_id", "label"])
plain = plain.merge(iris_plain, on=["pair_id", "label"])

print("Plain merged shape:", plain.shape)

# ============================================================
# 5. LOAD AND MERGE ENCRYPTED SCORES
# ============================================================
fp_ckks = load_score_csv(FP_CKKS, "fingerprint")
face_ckks = load_score_csv(FACE_CKKS, "face")
iris_ckks = load_score_csv(IRIS_CKKS, "iris")

enc = fp_ckks.merge(face_ckks, on=["pair_id", "label"])
enc = enc.merge(iris_ckks, on=["pair_id", "label"])

print("Encrypted merged shape:", enc.shape)

# ============================================================
# 6. RUN OPTIMIZED FUSION
# ============================================================
plain_metrics = fuse_three(plain[["pair_id", "label", "fingerprint", "face", "iris"]], "optimized_plain_all_three")
enc_metrics = fuse_three(enc[["pair_id", "label", "fingerprint", "face", "iris"]], "optimized_enc_all_three")

# ============================================================
# 7. SUMMARY
# ============================================================
summary = pd.DataFrame([
    {
        "fusion_name": "optimized_plain_all_three",
        **plain_metrics
    },
    {
        "fusion_name": "optimized_enc_all_three",
        **enc_metrics
    }
])

summary.to_csv(os.path.join(SAVE_DIR, "optimized_fusion_summary.csv"), index=False)

print("\n================ OPTIMIZED FUSION SUMMARY ================")
print(summary[[
    "fusion_name", "eer", "roc_auc", "accuracy",
    "tar_at_far_1_percent", "tar_at_far_0_1_percent"
]])
print("\n✅ Saved:", os.path.join(SAVE_DIR, "optimized_fusion_summary.csv"))
print("✅ DONE")

# ================= NOTEBOOK CELL 49 =================
# ============================================================
# VALIDATION WEIGHT SEARCH FOR:
# 1) Face + Fingerprint
# 2) Face + Iris
# 3) Fingerprint + Iris
# 4) Face + Fingerprint + Iris
# - Uses validation scores only
# - Saves best weights for each fusion case
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

SAVE_DIR = "/kaggle/working/fusion_weight_search_all"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 1. VALIDATION SCORE FILES
# ============================================================
FP_VAL = "/kaggle/working/fingerprint_training_balanced/best_val_scores.csv"
FACE_VAL = "/kaggle/working/face_training_balanced/best_val_scores.csv"
IRIS_VAL = "/kaggle/working/iris_training_balanced/best_val_scores.csv"

# ============================================================
# 2. HELPERS
# ============================================================
def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = thresholds[idx]
    return float(eer), float(thr)

def minmax_norm(x):
    x = np.asarray(x, dtype=np.float64)
    mn, mx = x.min(), x.max()
    if abs(mx - mn) < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def load_score_csv(path, name):
    df = pd.read_csv(path)
    df = df[["pair_id", "label", "score"]].copy()
    df = df.rename(columns={"score": name})
    return df

# ============================================================
# 3. LOAD + MERGE
# ============================================================
fp = load_score_csv(FP_VAL, "fingerprint")
face = load_score_csv(FACE_VAL, "face")
iris = load_score_csv(IRIS_VAL, "iris")

df = fp.merge(face, on=["pair_id", "label"])
df = df.merge(iris, on=["pair_id", "label"])

print("Merged val shape:", df.shape)

df["fingerprint_n"] = minmax_norm(df["fingerprint"].values)
df["face_n"] = minmax_norm(df["face"].values)
df["iris_n"] = minmax_norm(df["iris"].values)

labels = df["label"].values

# ============================================================
# 4. TWO-MODALITY SEARCH
# ============================================================
def search_two_modality(df, col1, col2, grid_step=0.05):
    results = []
    grid = np.arange(0.0, 1.01, grid_step)

    for w1 in grid:
        w2 = 1.0 - w1

        fused = (
            w1 * df[col1].values +
            w2 * df[col2].values
        )

        eer, thr = compute_eer(fused, labels)
        results.append([w1, w2, eer, thr])

    results_df = pd.DataFrame(
        results,
        columns=[f"w_{col1}", f"w_{col2}", "eer", "threshold"]
    ).sort_values("eer").reset_index(drop=True)

    return results_df

# ============================================================
# 5. THREE-MODALITY SEARCH
# ============================================================
def search_three_modality(df, c1, c2, c3, grid_step=0.05):
    results = []
    grid = np.arange(0.0, 1.01, grid_step)

    for w1 in grid:
        for w2 in grid:
            for w3 in grid:
                s = w1 + w2 + w3
                if abs(s - 1.0) > 1e-9:
                    continue

                fused = (
                    w1 * df[c1].values +
                    w2 * df[c2].values +
                    w3 * df[c3].values
                )

                eer, thr = compute_eer(fused, labels)
                results.append([w1, w2, w3, eer, thr])

    results_df = pd.DataFrame(
        results,
        columns=[f"w_{c1}", f"w_{c2}", f"w_{c3}", "eer", "threshold"]
    ).sort_values("eer").reset_index(drop=True)

    return results_df

# ============================================================
# 6. RUN SEARCHES
# ============================================================
# 2-modality
ff_results = search_two_modality(df, "face_n", "fingerprint_n")
fi_results = search_two_modality(df, "face_n", "iris_n")
fpi_results = search_two_modality(df, "fingerprint_n", "iris_n")

# 3-modality
all3_results = search_three_modality(df, "fingerprint_n", "face_n", "iris_n")

# ============================================================
# 7. SAVE ALL RESULTS
# ============================================================
ff_results.to_csv(os.path.join(SAVE_DIR, "face_fingerprint_weight_search.csv"), index=False)
fi_results.to_csv(os.path.join(SAVE_DIR, "face_iris_weight_search.csv"), index=False)
fpi_results.to_csv(os.path.join(SAVE_DIR, "fingerprint_iris_weight_search.csv"), index=False)
all3_results.to_csv(os.path.join(SAVE_DIR, "all_three_weight_search.csv"), index=False)

best_ff = ff_results.iloc[0].to_dict()
best_fi = fi_results.iloc[0].to_dict()
best_fpi = fpi_results.iloc[0].to_dict()
best_all3 = all3_results.iloc[0].to_dict()

with open(os.path.join(SAVE_DIR, "best_face_fingerprint_weights.json"), "w") as f:
    json.dump(best_ff, f, indent=4)

with open(os.path.join(SAVE_DIR, "best_face_iris_weights.json"), "w") as f:
    json.dump(best_fi, f, indent=4)

with open(os.path.join(SAVE_DIR, "best_fingerprint_iris_weights.json"), "w") as f:
    json.dump(best_fpi, f, indent=4)

with open(os.path.join(SAVE_DIR, "best_all_three_weights.json"), "w") as f:
    json.dump(best_all3, f, indent=4)

# ============================================================
# 8. PRINT RESULTS
# ============================================================
print("\n================ BEST 2-MODALITY WEIGHTS ================")
print("\nFace + Fingerprint")
print(ff_results.head(10))
print("\nBest:", best_ff)

print("\nFace + Iris")
print(fi_results.head(10))
print("\nBest:", best_fi)

print("\nFingerprint + Iris")
print(fpi_results.head(10))
print("\nBest:", best_fpi)

print("\n================ BEST 3-MODALITY WEIGHTS ================")
print(all3_results.head(10))
print("\nBest:", best_all3)

print("\n✅ Saved all weight search outputs to:", SAVE_DIR)

# ================= NOTEBOOK CELL 50 =================
# ============================================================
# FINAL MULTIMODAL + TRIMODAL FUSION
# - Uses optimized validation weights
# - Runs BOTH plain and encrypted-domain score fusion
# - Covers:
#     1) Face + Fingerprint
#     2) Face + Iris
#     3) Fingerprint + Iris
#     4) Face + Fingerprint + Iris
# ============================================================

import os
import json
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

# ============================================================
# 1. PATHS
# ============================================================
SAVE_DIR = "/kaggle/working/final_fusion_all_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- PLAIN TEST SCORES ----------
FP_PLAIN = "/kaggle/working/fingerprint_balanced_test_outputs/fingerprint_test_scores_balanced.csv"
FACE_PLAIN = "/kaggle/working/face_pretrained_test_outputs/face_test_scores_pretrained_balanced.csv"
IRIS_PLAIN = "/kaggle/working/iris_balanced_test_outputs/iris_test_scores_balanced.csv"

# ---------- ENCRYPTED TEST SCORES ----------
FP_CKKS = "/kaggle/working/fingerprint_ckks_outputs/fingerprint_test_scores_ckks.csv"
FACE_CKKS = "/kaggle/working/face_ckks_outputs/face_test_scores_ckks.csv"
IRIS_CKKS = "/kaggle/working/iris_ckks_outputs/iris_test_scores_ckks.csv"

# ============================================================
# 2. OPTIMIZED WEIGHTS FROM VALIDATION SEARCH
# ============================================================
# Face + Fingerprint
W_FF_FACE = 0.60
W_FF_FP   = 0.40

# Face + Iris
W_FI_FACE = 0.45
W_FI_IRIS = 0.55

# Fingerprint + Iris
W_FPI_FP   = 0.35
W_FPI_IRIS = 0.65

# All three
W_ALL_FP   = 0.30
W_ALL_FACE = 0.35
W_ALL_IRIS = 0.35

# ============================================================
# 3. HELPERS
# ============================================================
def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = thresholds[idx]
    return float(eer), float(thr), fpr, tpr, thresholds

def tar_at_far(labels, scores, target_far=0.01):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmin(np.abs(fpr - target_far))
    return float(tpr[idx]), float(fpr[idx]), float(thresholds[idx])

def minmax_norm(x):
    x = np.asarray(x, dtype=np.float64)
    mn, mx = x.min(), x.max()
    if abs(mx - mn) < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def load_score_csv(path, name):
    df = pd.read_csv(path)
    df = df[["pair_id", "label", "score"]].copy()
    df = df.rename(columns={"score": name})
    return df

def evaluate_scores(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

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

    extra = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "cm": cm
    }
    return metrics, extra

def show_and_save_curves(labels, scores, metrics, extra, prefix):
    fpr = extra["fpr"]
    tpr = extra["tpr"]
    cm = extra["cm"]
    thr = metrics["threshold_at_eer"]

    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    # ROC
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{prefix}_roc_curve.png"), dpi=200)
    plt.close()

    # score distribution
    plt.figure(figsize=(7,5))
    plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
    plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
    plt.axvline(thr, linestyle="--", label=f"Thr@EER={thr:.4f}")
    plt.xlabel("Fusion Score")
    plt.ylabel("Density")
    plt.title(f"{prefix} Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,5))
    plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
    plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
    plt.axvline(thr, linestyle="--", label=f"Thr@EER={thr:.4f}")
    plt.xlabel("Fusion Score")
    plt.ylabel("Density")
    plt.title(f"{prefix} Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{prefix}_score_distribution.png"), dpi=200)
    plt.close()

    # confusion matrix
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{prefix} Confusion Matrix")
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

    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{prefix} Confusion Matrix")
    plt.colorbar()
    plt.xticks(ticks, ["Impostor", "Genuine"])
    plt.yticks(ticks, ["Impostor", "Genuine"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{prefix}_confusion_matrix.png"), dpi=200)
    plt.close()

    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": extra["thresholds"]
    })
    roc_df.to_csv(os.path.join(SAVE_DIR, f"{prefix}_roc_points.csv"), index=False)

def fuse_two(df, col1, col2, w1, w2, prefix):
    work = df.copy()

    work[col1 + "_n"] = minmax_norm(work[col1].values)
    work[col2 + "_n"] = minmax_norm(work[col2].values)

    work["fusion_score"] = (
        w1 * work[col1 + "_n"].values +
        w2 * work[col2 + "_n"].values
    )

    labels = work["label"].values
    scores = work["fusion_score"].values

    metrics, extra = evaluate_scores(labels, scores)

    work.to_csv(os.path.join(SAVE_DIR, f"{prefix}_scores.csv"), index=False)

    with open(os.path.join(SAVE_DIR, f"{prefix}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n🔥 {prefix}")
    print(f"Weights          : {col1}={w1}, {col2}={w2}")
    print(f"EER              : {metrics['eer']*100:.2f}%")
    print(f"Threshold@EER    : {metrics['threshold_at_eer']:.4f}")
    print(f"ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Precision        : {metrics['precision']:.4f}")
    print(f"Recall           : {metrics['recall']:.4f}")
    print(f"F1-score         : {metrics['f1_score']:.4f}")
    print(f"FAR              : {metrics['far_at_eer_threshold']:.4f}")
    print(f"FRR              : {metrics['frr_at_eer_threshold']:.4f}")
    print(f"TAR@FAR=1%       : {metrics['tar_at_far_1_percent']:.4f}")
    print(f"TAR@FAR=0.1%     : {metrics['tar_at_far_0_1_percent']:.4f}")

    show_and_save_curves(labels, scores, metrics, extra, prefix)
    return metrics

def fuse_three(df, w_fp, w_face, w_iris, prefix):
    work = df.copy()

    work["fingerprint_n"] = minmax_norm(work["fingerprint"].values)
    work["face_n"] = minmax_norm(work["face"].values)
    work["iris_n"] = minmax_norm(work["iris"].values)

    work["fusion_score"] = (
        w_fp   * work["fingerprint_n"].values +
        w_face * work["face_n"].values +
        w_iris * work["iris_n"].values
    )

    labels = work["label"].values
    scores = work["fusion_score"].values

    metrics, extra = evaluate_scores(labels, scores)

    work.to_csv(os.path.join(SAVE_DIR, f"{prefix}_scores.csv"), index=False)

    with open(os.path.join(SAVE_DIR, f"{prefix}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n🔥 {prefix}")
    print(f"Weights          : FP={w_fp}, FACE={w_face}, IRIS={w_iris}")
    print(f"EER              : {metrics['eer']*100:.2f}%")
    print(f"Threshold@EER    : {metrics['threshold_at_eer']:.4f}")
    print(f"ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Precision        : {metrics['precision']:.4f}")
    print(f"Recall           : {metrics['recall']:.4f}")
    print(f"F1-score         : {metrics['f1_score']:.4f}")
    print(f"FAR              : {metrics['far_at_eer_threshold']:.4f}")
    print(f"FRR              : {metrics['frr_at_eer_threshold']:.4f}")
    print(f"TAR@FAR=1%       : {metrics['tar_at_far_1_percent']:.4f}")
    print(f"TAR@FAR=0.1%     : {metrics['tar_at_far_0_1_percent']:.4f}")

    show_and_save_curves(labels, scores, metrics, extra, prefix)
    return metrics

# ============================================================
# 4. LOAD PLAIN SCORES
# ============================================================
fp_plain = load_score_csv(FP_PLAIN, "fingerprint")
face_plain = load_score_csv(FACE_PLAIN, "face")
iris_plain = load_score_csv(IRIS_PLAIN, "iris")

plain = fp_plain.merge(face_plain, on=["pair_id", "label"])
plain = plain.merge(iris_plain, on=["pair_id", "label"])
print("Plain merged shape:", plain.shape)

# ============================================================
# 5. LOAD ENCRYPTED SCORES
# ============================================================
fp_ckks = load_score_csv(FP_CKKS, "fingerprint")
face_ckks = load_score_csv(FACE_CKKS, "face")
iris_ckks = load_score_csv(IRIS_CKKS, "iris")

enc = fp_ckks.merge(face_ckks, on=["pair_id", "label"])
enc = enc.merge(iris_ckks, on=["pair_id", "label"])
print("Encrypted merged shape:", enc.shape)

# ============================================================
# 6. RUN PLAIN FUSIONS
# ============================================================
plain_results = {}

plain_results["plain_face_fingerprint_opt"] = fuse_two(
    plain[["pair_id", "label", "face", "fingerprint"]],
    col1="face",
    col2="fingerprint",
    w1=W_FF_FACE,
    w2=W_FF_FP,
    prefix="plain_face_fingerprint_opt"
)

plain_results["plain_face_iris_opt"] = fuse_two(
    plain[["pair_id", "label", "face", "iris"]],
    col1="face",
    col2="iris",
    w1=W_FI_FACE,
    w2=W_FI_IRIS,
    prefix="plain_face_iris_opt"
)

plain_results["plain_fingerprint_iris_opt"] = fuse_two(
    plain[["pair_id", "label", "fingerprint", "iris"]],
    col1="fingerprint",
    col2="iris",
    w1=W_FPI_FP,
    w2=W_FPI_IRIS,
    prefix="plain_fingerprint_iris_opt"
)

plain_results["plain_all_three_opt"] = fuse_three(
    plain[["pair_id", "label", "fingerprint", "face", "iris"]],
    w_fp=W_ALL_FP,
    w_face=W_ALL_FACE,
    w_iris=W_ALL_IRIS,
    prefix="plain_all_three_opt"
)

# ============================================================
# 7. RUN ENCRYPTED FUSIONS
# ============================================================
enc_results = {}

enc_results["enc_face_fingerprint_opt"] = fuse_two(
    enc[["pair_id", "label", "face", "fingerprint"]],
    col1="face",
    col2="fingerprint",
    w1=W_FF_FACE,
    w2=W_FF_FP,
    prefix="enc_face_fingerprint_opt"
)

enc_results["enc_face_iris_opt"] = fuse_two(
    enc[["pair_id", "label", "face", "iris"]],
    col1="face",
    col2="iris",
    w1=W_FI_FACE,
    w2=W_FI_IRIS,
    prefix="enc_face_iris_opt"
)

enc_results["enc_fingerprint_iris_opt"] = fuse_two(
    enc[["pair_id", "label", "fingerprint", "iris"]],
    col1="fingerprint",
    col2="iris",
    w1=W_FPI_FP,
    w2=W_FPI_IRIS,
    prefix="enc_fingerprint_iris_opt"
)

enc_results["enc_all_three_opt"] = fuse_three(
    enc[["pair_id", "label", "fingerprint", "face", "iris"]],
    w_fp=W_ALL_FP,
    w_face=W_ALL_FACE,
    w_iris=W_ALL_IRIS,
    prefix="enc_all_three_opt"
)

# ============================================================
# 8. SUMMARY TABLE
# ============================================================
rows = []

for name, met in plain_results.items():
    rows.append([
        name,
        met["eer"],
        met["roc_auc"],
        met["accuracy"],
        met["tar_at_far_1_percent"],
        met["tar_at_far_0_1_percent"]
    ])

for name, met in enc_results.items():
    rows.append([
        name,
        met["eer"],
        met["roc_auc"],
        met["accuracy"],
        met["tar_at_far_1_percent"],
        met["tar_at_far_0_1_percent"]
    ])

summary_df = pd.DataFrame(
    rows,
    columns=[
        "fusion_name",
        "eer",
        "roc_auc",
        "accuracy",
        "tar_at_far_1_percent",
        "tar_at_far_0_1_percent"
    ]
)

summary_df = summary_df.sort_values("eer").reset_index(drop=True)
summary_df.to_csv(os.path.join(SAVE_DIR, "final_fusion_summary.csv"), index=False)

print("\n================ FINAL FUSION SUMMARY ================")
print(summary_df)
print("\n✅ Saved:", os.path.join(SAVE_DIR, "final_fusion_summary.csv"))
print("✅ DONE")

# ================= NOTEBOOK CELL 88 =================
# ============================================================
# TRUE CKKS FUSION: BIMODAL + TRIMODAL
# Uses true CKKS score CSVs from fingerprint, face, iris
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

# ============================================================
# 1. PATHS
# ============================================================

FP_PATH   = "/kaggle/working/fingerprint_true_ckks_outputs/fingerprint_test_scores_true_ckks.csv"
FACE_PATH = "/kaggle/working/face_true_ckks_outputs/face_test_scores_true_ckks.csv"
IRIS_PATH = "/kaggle/working/iris_true_ckks_outputs/iris_test_scores_true_ckks.csv"

SAVE_ROOT = "/kaggle/working/fusion_true_ckks_outputs"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ============================================================
# 2. CONFIG
# Use your validated protected-fusion weights
# ============================================================

DEFAULT_WEIGHTS = {
    "face": 0.30,
    "fingerprint": 0.20,
    "iris": 0.50
}

# You can also keep your earlier bimodal tuned weights:
PAIR_WEIGHTS = {
    "face_fingerprint": {"face": 0.60, "fingerprint": 0.40},
    "face_iris": {"face": 0.375, "iris": 0.625},
    "fingerprint_iris": {"fingerprint": 0.2857, "iris": 0.7143},
    "all_three": {"face": 0.30, "fingerprint": 0.20, "iris": 0.50},
}

# ============================================================
# 3. HELPERS
# ============================================================

def normalize_scores(x):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def compute_metrics(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(fpr[eer_idx])
    thr = float(thresholds[eer_idx])

    if not np.isfinite(thr):
        thr = 0.5

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

def load_score_file(path, modality_name):
    df = pd.read_csv(path)
    keep_cols = ["pair_id", "subject1", "subject2", "idx1", "idx2", "label", "score"]
    df = df[keep_cols].copy()
    df = df.rename(columns={"score": f"score_{modality_name}"})
    return df

def save_plots(df, metrics, out_dir, title_prefix):
    os.makedirs(out_dir, exist_ok=True)

    fpr = metrics["fpr"]
    tpr = metrics["tpr"]
    thr = metrics["threshold"]
    auc_val = metrics["auc"]
    cm = metrics["cm"]

    # ROC
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Score distribution
    genuine = df[df["label"] == 1]["fused_score"].values
    impostor = df[df["label"] == 0]["fused_score"].values

    plt.figure(figsize=(7, 5))
    plt.hist(genuine, bins=50, alpha=0.6, density=True, label="Genuine")
    plt.hist(impostor, bins=50, alpha=0.6, density=True, label="Impostor")
    plt.axvline(thr, linestyle="--", label=f"Threshold = {thr:.4f}")
    plt.xlabel("Fused Score")
    plt.ylabel("Density")
    plt.title(f"{title_prefix} Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "score_distribution.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Confusion matrix
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{title_prefix} Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Impostor", "Genuine"])
    plt.yticks([0, 1], ["Impostor", "Genuine"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================
# 4. LOAD SCORE FILES
# ============================================================

fp_df = load_score_file(FP_PATH, "fingerprint")
face_df = load_score_file(FACE_PATH, "face")
iris_df = load_score_file(IRIS_PATH, "iris")

merge_keys = ["pair_id", "subject1", "subject2", "idx1", "idx2", "label"]

all_df = face_df.merge(fp_df, on=merge_keys).merge(iris_df, on=merge_keys)

# normalize per modality
all_df["score_face_norm"] = normalize_scores(all_df["score_face"])
all_df["score_fingerprint_norm"] = normalize_scores(all_df["score_fingerprint"])
all_df["score_iris_norm"] = normalize_scores(all_df["score_iris"])

print(all_df.head())

# ============================================================
# 5. FUSION CONFIGS
# ============================================================

fusion_sets = {
    "face_fingerprint": ["face", "fingerprint"],
    "face_iris": ["face", "iris"],
    "fingerprint_iris": ["fingerprint", "iris"],
    "all_three": ["face", "fingerprint", "iris"]
}

summary_rows = []

# ============================================================
# 6. RUN FUSIONS
# ============================================================

for fusion_name, mods in fusion_sets.items():
    df = all_df.copy()

    # weights
    weight_dict = PAIR_WEIGHTS[fusion_name]
    weights = np.array([weight_dict[m] for m in mods], dtype=np.float32)
    weights = weights / weights.sum()

    score_cols = [f"score_{m}_norm" for m in mods]
    score_mat = df[score_cols].values.astype(np.float32)

    fused = np.sum(score_mat * weights.reshape(1, -1), axis=1)
    df["fused_score"] = fused

    y_true = df["label"].values.astype(int)
    metrics = compute_metrics(y_true, df["fused_score"].values)

    out_dir = os.path.join(SAVE_ROOT, fusion_name)
    os.makedirs(out_dir, exist_ok=True)

    # save fused scores
    fused_csv = os.path.join(out_dir, f"{fusion_name}_scores.csv")
    df.to_csv(fused_csv, index=False)

    # save metrics
    summary = {
        "mode": "true_ckks_fusion",
        "fusion_name": fusion_name,
        "modalities": ",".join(mods),
        "weights": ",".join([f"{m}:{w:.4f}" for m, w in zip(mods, weights)]),
        "eer": metrics["eer"],
        "threshold": metrics["threshold"],
        "auc": metrics["auc"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
    }

    with open(os.path.join(out_dir, f"{fusion_name}_metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)

    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, f"{fusion_name}_result_row.csv"),
        index=False
    )

    roc_points_df = pd.DataFrame({
        "fpr": metrics["fpr"],
        "tpr": metrics["tpr"],
        "threshold": metrics["thresholds"]
    })
    roc_points_df.to_csv(os.path.join(out_dir, "roc_points.csv"), index=False)

    save_plots(df, metrics, out_dir, f"TRUE CKKS Fusion: {fusion_name}")

    summary_rows.append(summary)

    print(f"\nDone: {fusion_name}")
    print(summary)

# ============================================================
# 7. FINAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(SAVE_ROOT, "fusion_true_ckks_summary.csv")
summary_df.to_csv(summary_csv, index=False)

print("\n===== FINAL TRUE CKKS FUSION SUMMARY =====")
print(summary_df)
print("\nSaved:", summary_csv)
