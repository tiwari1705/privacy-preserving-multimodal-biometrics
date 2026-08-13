"""Extracted from notebook3e7611700a(1).ipynb.
Cells preserved in original execution order.
Kaggle paths are intentionally preserved; replace them for local execution.
"""

# ================= NOTEBOOK CELL 79 =================
# ============================================================
# REVOCABILITY + CROSS-KEY UNLINKABILITY EXPERIMENT
# Fingerprint version
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

INPUT_ROOT = "/kaggle/input/datasets/radhe11/backup/kaggle/working"
SAVE_DIR = "/kaggle/working/revocability_fingerprint"
os.makedirs(SAVE_DIR, exist_ok=True)

EMB_PATH = os.path.join(INPUT_ROOT, "fingerprint_balanced_test_outputs", "fingerprint_test_embeddings.npy")
META_PATH = os.path.join(INPUT_ROOT, "fingerprint_balanced_test_outputs", "fingerprint_test_embeddings_meta.csv")
PAIRS_PATH = os.path.join(INPUT_ROOT, "common_pairs_balanced", "test_pairs_common_balanced.csv")

KEYS = [11, 77, 123]
OUT_DIM = 512
BINARY = False

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
emb = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH)
pairs = pd.read_csv(PAIRS_PATH)

meta = meta.copy()
meta["global_index"] = np.arange(len(meta))
meta = meta.sort_values(["subject", "image_path"]).reset_index(drop=True)
meta["local_idx"] = meta.groupby("subject").cumcount()

index_map = {
    (int(r.subject), int(r.local_idx)): int(r.global_index)
    for _, r in meta.iterrows()
}

# ------------------------------------------------------------
# Cancellable transform
# ------------------------------------------------------------
def cancellable_transform(embeddings, key=11, out_dim=512, binary=False):
    rng = np.random.default_rng(seed=key)
    R = rng.standard_normal((embeddings.shape[1], out_dim))
    proj = embeddings @ R
    proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
    if binary:
        return (proj > 0).astype(np.float32)
    return proj.astype(np.float32)

templates = {}
for k in KEYS:
    templates[k] = cancellable_transform(emb, key=k, out_dim=OUT_DIM, binary=BINARY)

# ------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------
def evaluate_pairing(probe_templates, gallery_templates, pairs_df, name="same_key"):
    scores = []

    for _, row in pairs_df.iterrows():
        g1 = index_map[(int(row["subject1"]), int(row["idx1"]))]
        g2 = index_map[(int(row["subject2"]), int(row["idx2"]))]

        s = cosine_similarity(
            probe_templates[g1].reshape(1, -1),
            gallery_templates[g2].reshape(1, -1)
        )[0][0]
        scores.append(float(s))

    out = pairs_df.copy()
    out["score"] = scores

    y = out["label"].values.astype(int)
    s = out["score"].values.astype(float)

    fpr, tpr, th = roc_curve(y, s)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(fpr[eer_idx])
    thr = float(th[eer_idx])
    auc = float(roc_auc_score(y, s))

    if not np.isfinite(thr):
        thr = 0.5

    y_pred = (s >= thr).astype(int)
    acc = float(accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred))

    return {
        "experiment": name,
        "eer": eer,
        "auc": auc,
        "accuracy": acc,
        "f1": f1,
        "threshold": thr,
        "scores_df": out
    }

# ------------------------------------------------------------
# Same-key performance
# ------------------------------------------------------------
results = []

for k in KEYS:
    r = evaluate_pairing(
        templates[k],
        templates[k],
        pairs,
        name=f"same_key_{k}"
    )
    results.append({
        "experiment": r["experiment"],
        "eer": r["eer"],
        "auc": r["auc"],
        "accuracy": r["accuracy"],
        "f1": r["f1"]
    })

# ------------------------------------------------------------
# Cross-key unlinkability performance
# gallery = key11, probe = key77 etc.
# ------------------------------------------------------------
cross_key_configs = [
    (11, 77),
    (11, 123),
    (77, 123)
]

for k1, k2 in cross_key_configs:
    r = evaluate_pairing(
        templates[k1],
        templates[k2],
        pairs,
        name=f"cross_key_{k1}_vs_{k2}"
    )
    results.append({
        "experiment": r["experiment"],
        "eer": r["eer"],
        "auc": r["auc"],
        "accuracy": r["accuracy"],
        "f1": r["f1"]
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(SAVE_DIR, "revocability_results.csv"), index=False)

print(results_df)

# ------------------------------------------------------------
# Template-level cross-key similarity analysis
# Compare same sample under different keys
# ------------------------------------------------------------
sim_rows = []

for i in range(len(emb)):
    for k1, k2 in cross_key_configs:
        sim = cosine_similarity(
            templates[k1][i].reshape(1, -1),
            templates[k2][i].reshape(1, -1)
        )[0][0]
        sim_rows.append({
            "sample_index": i,
            "key_pair": f"{k1}_vs_{k2}",
            "similarity": float(sim)
        })

sim_df = pd.DataFrame(sim_rows)
sim_df.to_csv(os.path.join(SAVE_DIR, "cross_key_template_similarity.csv"), index=False)

# ------------------------------------------------------------
# Plot 1: same-key vs cross-key AUC
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.bar(results_df["experiment"], results_df["auc"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("AUC")
plt.title("Revocability Experiment: Same-Key vs Cross-Key AUC")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "revocability_auc_bar.png"), dpi=300)
plt.show()

# ------------------------------------------------------------
# Plot 2: same-key vs cross-key EER
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.bar(results_df["experiment"], results_df["eer"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("EER")
plt.title("Revocability Experiment: Same-Key vs Cross-Key EER")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "revocability_eer_bar.png"), dpi=300)
plt.show()

# ------------------------------------------------------------
# Plot 3: cross-key template similarity distribution
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
for kp in sim_df["key_pair"].unique():
    vals = sim_df[sim_df["key_pair"] == kp]["similarity"].values
    plt.hist(vals, bins=50, alpha=0.5, density=True, label=kp)

plt.xlabel("Similarity")
plt.ylabel("Density")
plt.title("Cross-Key Template Similarity Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "cross_key_similarity_distribution.png"), dpi=300)
plt.show()

# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------
summary = {
    "modality": "fingerprint",
    "out_dim": OUT_DIM,
    "binary": BINARY,
    "keys": KEYS,
    "interpretation": "Same-key should remain strong; cross-key should degrade if templates are revocable/unlinkable."
}

with open(os.path.join(SAVE_DIR, "revocability_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print("\nSaved all revocability outputs in:", SAVE_DIR)

# ================= NOTEBOOK CELL 80 =================
# ============================================================
# REVOCABILITY + CROSS-KEY UNLINKABILITY EXPERIMENT
# FACE VERSION
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
INPUT_ROOT = "/kaggle/input/datasets/radhe11/backup/kaggle/working"
SAVE_DIR = "/kaggle/working/revocability_face"
os.makedirs(SAVE_DIR, exist_ok=True)

EMB_PATH = os.path.join(INPUT_ROOT, "face_pretrained_embeddings", "face_test_embeddings_pretrained.npy")
META_PATH = os.path.join(INPUT_ROOT, "face_pretrained_embeddings", "face_test_embeddings_meta_pretrained.csv")
PAIRS_PATH = os.path.join(INPUT_ROOT, "common_pairs_balanced", "test_pairs_common_balanced.csv")

KEYS = [11, 77, 123]
OUT_DIM = 512
BINARY = False

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------
emb = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH)
pairs = pd.read_csv(PAIRS_PATH)

print("Embeddings:", emb.shape)
print("Meta:", meta.shape)
print("Pairs:", pairs.shape)
print("Meta columns:", list(meta.columns))
print("Pairs columns:", list(pairs.columns))

# ------------------------------------------------------------
# BUILD CORRECT MAPPING
# (subject, local_idx) -> global embedding row
# ------------------------------------------------------------
meta = meta.copy()
meta["global_index"] = np.arange(len(meta))
meta = meta.sort_values(["subject", "image_path"]).reset_index(drop=True)
meta["local_idx"] = meta.groupby("subject").cumcount()

index_map = {
    (int(r.subject), int(r.local_idx)): int(r.global_index)
    for _, r in meta.iterrows()
}

print(meta[["subject", "local_idx", "global_index"]].head(10))

# ------------------------------------------------------------
# CANCELLABLE TRANSFORM
# ------------------------------------------------------------
def cancellable_transform(embeddings, key=11, out_dim=512, binary=False):
    rng = np.random.default_rng(seed=key)
    R = rng.standard_normal((embeddings.shape[1], out_dim))
    proj = embeddings @ R
    proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
    if binary:
        return (proj > 0).astype(np.float32)
    return proj.astype(np.float32)

templates = {}
for k in KEYS:
    templates[k] = cancellable_transform(emb, key=k, out_dim=OUT_DIM, binary=BINARY)

# ------------------------------------------------------------
# EVALUATION HELPER
# ------------------------------------------------------------
def evaluate_pairing(probe_templates, gallery_templates, pairs_df, name="same_key"):
    scores = []

    for _, row in pairs_df.iterrows():
        g1 = index_map[(int(row["subject1"]), int(row["idx1"]))]
        g2 = index_map[(int(row["subject2"]), int(row["idx2"]))]

        s = cosine_similarity(
            probe_templates[g1].reshape(1, -1),
            gallery_templates[g2].reshape(1, -1)
        )[0][0]
        scores.append(float(s))

    out = pairs_df.copy()
    out["score"] = scores

    y = out["label"].values.astype(int)
    s = out["score"].values.astype(float)

    fpr, tpr, th = roc_curve(y, s)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(fpr[eer_idx])
    thr = float(th[eer_idx])
    auc = float(roc_auc_score(y, s))

    if not np.isfinite(thr):
        thr = 0.5

    y_pred = (s >= thr).astype(int)
    acc = float(accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred))

    return {
        "experiment": name,
        "eer": eer,
        "auc": auc,
        "accuracy": acc,
        "f1": f1,
        "threshold": thr,
        "scores_df": out
    }

# ------------------------------------------------------------
# SAME-KEY PERFORMANCE
# ------------------------------------------------------------
results = []

for k in KEYS:
    r = evaluate_pairing(
        templates[k],
        templates[k],
        pairs,
        name=f"same_key_{k}"
    )
    results.append({
        "experiment": r["experiment"],
        "eer": r["eer"],
        "auc": r["auc"],
        "accuracy": r["accuracy"],
        "f1": r["f1"]
    })

# ------------------------------------------------------------
# CROSS-KEY UNLINKABILITY
# ------------------------------------------------------------
cross_key_configs = [
    (11, 77),
    (11, 123),
    (77, 123)
]

for k1, k2 in cross_key_configs:
    r = evaluate_pairing(
        templates[k1],
        templates[k2],
        pairs,
        name=f"cross_key_{k1}_vs_{k2}"
    )
    results.append({
        "experiment": r["experiment"],
        "eer": r["eer"],
        "auc": r["auc"],
        "accuracy": r["accuracy"],
        "f1": r["f1"]
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(SAVE_DIR, "revocability_face_results.csv"), index=False)

print("\n=== FACE REVOCABILITY RESULTS ===")
print(results_df)

# ------------------------------------------------------------
# TEMPLATE-LEVEL CROSS-KEY SIMILARITY
# ------------------------------------------------------------
sim_rows = []

for i in range(len(emb)):
    for k1, k2 in cross_key_configs:
        sim = cosine_similarity(
            templates[k1][i].reshape(1, -1),
            templates[k2][i].reshape(1, -1)
        )[0][0]
        sim_rows.append({
            "sample_index": i,
            "key_pair": f"{k1}_vs_{k2}",
            "similarity": float(sim)
        })

sim_df = pd.DataFrame(sim_rows)
sim_df.to_csv(os.path.join(SAVE_DIR, "cross_key_template_similarity_face.csv"), index=False)

# ------------------------------------------------------------
# PLOT 1: AUC BAR
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.bar(results_df["experiment"], results_df["auc"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("AUC")
plt.title("Face Revocability: Same-Key vs Cross-Key AUC")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "revocability_face_auc_bar.png"), dpi=300)
plt.show()

# ------------------------------------------------------------
# PLOT 2: EER BAR
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.bar(results_df["experiment"], results_df["eer"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("EER")
plt.title("Face Revocability: Same-Key vs Cross-Key EER")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "revocability_face_eer_bar.png"), dpi=300)
plt.show()

# ------------------------------------------------------------
# PLOT 3: SIMILARITY DISTRIBUTION
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
for kp in sim_df["key_pair"].unique():
    vals = sim_df[sim_df["key_pair"] == kp]["similarity"].values
    plt.hist(vals, bins=50, alpha=0.5, density=True, label=kp)

plt.xlabel("Similarity")
plt.ylabel("Density")
plt.title("Face Cross-Key Template Similarity Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "cross_key_similarity_distribution_face.png"), dpi=300)
plt.show()

# ------------------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------------------
summary = {
    "modality": "face",
    "out_dim": OUT_DIM,
    "binary": BINARY,
    "keys": KEYS,
    "interpretation": "Same-key should remain strong; cross-key should degrade if templates are revocable/unlinkable."
}

with open(os.path.join(SAVE_DIR, "revocability_face_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print("\nSaved all face revocability outputs in:", SAVE_DIR)

# ================= NOTEBOOK CELL 81 =================
# ============================================================
# REVOCABILITY + CROSS-KEY UNLINKABILITY EXPERIMENT
# IRIS VERSION
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
INPUT_ROOT = "/kaggle/input/datasets/radhe11/backup/kaggle/working"
SAVE_DIR = "/kaggle/working/revocability_iris"
os.makedirs(SAVE_DIR, exist_ok=True)

EMB_PATH = os.path.join(INPUT_ROOT, "iris_balanced_test_outputs", "iris_test_embeddings.npy")
META_PATH = os.path.join(INPUT_ROOT, "iris_balanced_test_outputs", "iris_test_embeddings_meta.csv")
PAIRS_PATH = os.path.join(INPUT_ROOT, "common_pairs_balanced", "test_pairs_common_balanced.csv")

KEYS = [11, 77, 123]
OUT_DIM = 512
BINARY = False

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------
emb = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH)
pairs = pd.read_csv(PAIRS_PATH)

print("Embeddings:", emb.shape)
print("Meta:", meta.shape)
print("Pairs:", pairs.shape)
print("Meta columns:", list(meta.columns))
print("Pairs columns:", list(pairs.columns))

# ------------------------------------------------------------
# BUILD CORRECT MAPPING
# ------------------------------------------------------------
meta = meta.copy()
meta["global_index"] = np.arange(len(meta))
meta = meta.sort_values(["subject", "image_path"]).reset_index(drop=True)
meta["local_idx"] = meta.groupby("subject").cumcount()

index_map = {
    (int(r.subject), int(r.local_idx)): int(r.global_index)
    for _, r in meta.iterrows()
}

print(meta[["subject", "local_idx", "global_index"]].head(10))

# ------------------------------------------------------------
# CANCELLABLE TRANSFORM
# ------------------------------------------------------------
def cancellable_transform(embeddings, key=11, out_dim=512, binary=False):
    rng = np.random.default_rng(seed=key)
    R = rng.standard_normal((embeddings.shape[1], out_dim))
    proj = embeddings @ R
    proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
    if binary:
        return (proj > 0).astype(np.float32)
    return proj.astype(np.float32)

templates = {}
for k in KEYS:
    templates[k] = cancellable_transform(emb, key=k, out_dim=OUT_DIM, binary=BINARY)

# ------------------------------------------------------------
# EVALUATION HELPER
# ------------------------------------------------------------
def evaluate_pairing(probe_templates, gallery_templates, pairs_df, name="same_key"):
    scores = []

    for _, row in pairs_df.iterrows():
        g1 = index_map[(int(row["subject1"]), int(row["idx1"]))]
        g2 = index_map[(int(row["subject2"]), int(row["idx2"]))]

        s = cosine_similarity(
            probe_templates[g1].reshape(1, -1),
            gallery_templates[g2].reshape(1, -1)
        )[0][0]
        scores.append(float(s))

    out = pairs_df.copy()
    out["score"] = scores

    y = out["label"].values.astype(int)
    s = out["score"].values.astype(float)

    fpr, tpr, th = roc_curve(y, s)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(fpr[eer_idx])
    thr = float(th[eer_idx])
    auc = float(roc_auc_score(y, s))

    if not np.isfinite(thr):
        thr = 0.5

    y_pred = (s >= thr).astype(int)
    acc = float(accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred))

    return {
        "experiment": name,
        "eer": eer,
        "auc": auc,
        "accuracy": acc,
        "f1": f1,
        "threshold": thr,
        "scores_df": out
    }

# ------------------------------------------------------------
# SAME-KEY PERFORMANCE
# ------------------------------------------------------------
results = []

for k in KEYS:
    r = evaluate_pairing(
        templates[k],
        templates[k],
        pairs,
        name=f"same_key_{k}"
    )
    results.append({
        "experiment": r["experiment"],
        "eer": r["eer"],
        "auc": r["auc"],
        "accuracy": r["accuracy"],
        "f1": r["f1"]
    })

# ------------------------------------------------------------
# CROSS-KEY UNLINKABILITY
# ------------------------------------------------------------
cross_key_configs = [
    (11, 77),
    (11, 123),
    (77, 123)
]

for k1, k2 in cross_key_configs:
    r = evaluate_pairing(
        templates[k1],
        templates[k2],
        pairs,
        name=f"cross_key_{k1}_vs_{k2}"
    )
    results.append({
        "experiment": r["experiment"],
        "eer": r["eer"],
        "auc": r["auc"],
        "accuracy": r["accuracy"],
        "f1": r["f1"]
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(SAVE_DIR, "revocability_iris_results.csv"), index=False)

print("\n=== IRIS REVOCABILITY RESULTS ===")
print(results_df)

# ------------------------------------------------------------
# TEMPLATE-LEVEL CROSS-KEY SIMILARITY
# ------------------------------------------------------------
sim_rows = []

for i in range(len(emb)):
    for k1, k2 in cross_key_configs:
        sim = cosine_similarity(
            templates[k1][i].reshape(1, -1),
            templates[k2][i].reshape(1, -1)
        )[0][0]
        sim_rows.append({
            "sample_index": i,
            "key_pair": f"{k1}_vs_{k2}",
            "similarity": float(sim)
        })

sim_df = pd.DataFrame(sim_rows)
sim_df.to_csv(os.path.join(SAVE_DIR, "cross_key_template_similarity_iris.csv"), index=False)

# ------------------------------------------------------------
# PLOT 1: AUC BAR
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.bar(results_df["experiment"], results_df["auc"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("AUC")
plt.title("Iris Revocability: Same-Key vs Cross-Key AUC")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "revocability_iris_auc_bar.png"), dpi=300)
plt.show()

# ------------------------------------------------------------
# PLOT 2: EER BAR
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.bar(results_df["experiment"], results_df["eer"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("EER")
plt.title("Iris Revocability: Same-Key vs Cross-Key EER")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "revocability_iris_eer_bar.png"), dpi=300)
plt.show()

# ------------------------------------------------------------
# PLOT 3: SIMILARITY DISTRIBUTION
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
for kp in sim_df["key_pair"].unique():
    vals = sim_df[sim_df["key_pair"] == kp]["similarity"].values
    plt.hist(vals, bins=50, alpha=0.5, density=True, label=kp)

plt.xlabel("Similarity")
plt.ylabel("Density")
plt.title("Iris Cross-Key Template Similarity Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "cross_key_similarity_distribution_iris.png"), dpi=300)
plt.show()

# ------------------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------------------
summary = {
    "modality": "iris",
    "out_dim": OUT_DIM,
    "binary": BINARY,
    "keys": KEYS,
    "interpretation": "Same-key should remain strong; cross-key should degrade if templates are revocable/unlinkable."
}

with open(os.path.join(SAVE_DIR, "revocability_iris_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print("\nSaved all iris revocability outputs in:", SAVE_DIR)

# ================= NOTEBOOK CELL 82 =================
# ============================================================
# SECURITY-UTILITY CURVE FOR FACE / FINGERPRINT / IRIS
# Runs multiple projection dimensions and binary/non-binary settings
# Saves per-modality results + plots + combined summary
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

INPUT_ROOT = "/kaggle/input/datasets/radhe11/backup/kaggle/working"
SAVE_ROOT = "/kaggle/working/security_utility_all_modalities"
os.makedirs(SAVE_ROOT, exist_ok=True)

KEY = 11
DIMS = [64, 128, 256, 512, 1024]
BINARY_OPTIONS = [True, False]

CONFIGS = {
    "fingerprint": {
        "emb_path": os.path.join(INPUT_ROOT, "fingerprint_balanced_test_outputs", "fingerprint_test_embeddings.npy"),
        "meta_path": os.path.join(INPUT_ROOT, "fingerprint_balanced_test_outputs", "fingerprint_test_embeddings_meta.csv"),
    },
    "face": {
        "emb_path": os.path.join(INPUT_ROOT, "face_pretrained_embeddings", "face_test_embeddings_pretrained.npy"),
        "meta_path": os.path.join(INPUT_ROOT, "face_pretrained_embeddings", "face_test_embeddings_meta_pretrained.csv"),
    },
    "iris": {
        "emb_path": os.path.join(INPUT_ROOT, "iris_balanced_test_outputs", "iris_test_embeddings.npy"),
        "meta_path": os.path.join(INPUT_ROOT, "iris_balanced_test_outputs", "iris_test_embeddings_meta.csv"),
    }
}

PAIRS_PATH = os.path.join(INPUT_ROOT, "common_pairs_balanced", "test_pairs_common_balanced.csv")
pairs = pd.read_csv(PAIRS_PATH)

def cancellable_transform(embeddings, key=11, out_dim=512, binary=False):
    rng = np.random.default_rng(seed=key)
    R = rng.standard_normal((embeddings.shape[1], out_dim))
    proj = embeddings @ R
    proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
    if binary:
        return (proj > 0).astype(np.float32)
    return proj.astype(np.float32)

def build_index_map(meta):
    meta = meta.copy()
    meta["global_index"] = np.arange(len(meta))
    meta = meta.sort_values(["subject", "image_path"]).reset_index(drop=True)
    meta["local_idx"] = meta.groupby("subject").cumcount()
    return {
        (int(r.subject), int(r.local_idx)): int(r.global_index)
        for _, r in meta.iterrows()
    }

def evaluate_templates(temp, pairs_df, index_map):
    scores = []

    for _, row in pairs_df.iterrows():
        g1 = index_map[(int(row["subject1"]), int(row["idx1"]))]
        g2 = index_map[(int(row["subject2"]), int(row["idx2"]))]

        s = cosine_similarity(
            temp[g1].reshape(1, -1),
            temp[g2].reshape(1, -1)
        )[0][0]
        scores.append(float(s))

    y = pairs_df["label"].values.astype(int)
    s = np.array(scores, dtype=np.float32)

    fpr, tpr, th = roc_curve(y, s)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(fpr[eer_idx])
    thr = float(th[eer_idx])
    auc = float(roc_auc_score(y, s))

    if not np.isfinite(thr):
        thr = 0.5

    y_pred = (s >= thr).astype(int)
    acc = float(accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred))

    return eer, auc, acc, f1

all_rows = []

for modality, cfg in CONFIGS.items():
    mod_save = os.path.join(SAVE_ROOT, modality)
    os.makedirs(mod_save, exist_ok=True)

    emb = np.load(cfg["emb_path"])
    meta = pd.read_csv(cfg["meta_path"])
    index_map = build_index_map(meta)

    rows = []

    for binary in BINARY_OPTIONS:
        for dim in DIMS:
            temp = cancellable_transform(emb, key=KEY, out_dim=dim, binary=binary)
            eer, auc, acc, f1 = evaluate_templates(temp, pairs, index_map)

            row = {
                "modality": modality,
                "dim": dim,
                "binary": binary,
                "eer": eer,
                "auc": auc,
                "accuracy": acc,
                "f1": f1
            }
            rows.append(row)
            all_rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(mod_save, f"{modality}_security_utility_results.csv"), index=False)
    print(f"\n=== {modality.upper()} SECURITY-UTILITY ===")
    print(df)

    # Accuracy vs Dimension
    plt.figure(figsize=(8, 5))
    for binary in BINARY_OPTIONS:
        sub = df[df["binary"] == binary]
        plt.plot(sub["dim"], sub["accuracy"], marker="o", label=f"binary={binary}")
    plt.xlabel("Projection Dimension")
    plt.ylabel("Accuracy")
    plt.title(f"{modality.capitalize()}: Accuracy vs Dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(mod_save, f"{modality}_accuracy_vs_dimension.png"), dpi=300)
    plt.show()

    # EER vs Dimension
    plt.figure(figsize=(8, 5))
    for binary in BINARY_OPTIONS:
        sub = df[df["binary"] == binary]
        plt.plot(sub["dim"], sub["eer"], marker="o", label=f"binary={binary}")
    plt.xlabel("Projection Dimension")
    plt.ylabel("EER")
    plt.title(f"{modality.capitalize()}: EER vs Dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(mod_save, f"{modality}_eer_vs_dimension.png"), dpi=300)
    plt.show()

    # AUC vs Dimension
    plt.figure(figsize=(8, 5))
    for binary in BINARY_OPTIONS:
        sub = df[df["binary"] == binary]
        plt.plot(sub["dim"], sub["auc"], marker="o", label=f"binary={binary}")
    plt.xlabel("Projection Dimension")
    plt.ylabel("AUC")
    plt.title(f"{modality.capitalize()}: AUC vs Dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(mod_save, f"{modality}_auc_vs_dimension.png"), dpi=300)
    plt.show()

all_df = pd.DataFrame(all_rows)
all_df.to_csv(os.path.join(SAVE_ROOT, "all_modalities_security_utility_results.csv"), index=False)

with open(os.path.join(SAVE_ROOT, "security_utility_summary.json"), "w") as f:
    json.dump({
        "key": KEY,
        "dims": DIMS,
        "binary_options": BINARY_OPTIONS
    }, f, indent=4)

print("\nSaved all security-utility outputs in:", SAVE_ROOT)

# ================= NOTEBOOK CELL 108 =================
# ============================================================
# IRREVERSIBILITY ANALYSIS (BEST PRACTICAL VERSION)
# Distance correlation + NN overlap + reconstruction baseline
# Recommended protected config
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ============================================================
# 1. CONFIG
# ============================================================

MODALITY = "fingerprint"   # change to "face" or "iris" later
KEY = 11

CONFIGS = {
    "fingerprint": {
        "emb_path": "/kaggle/input/datasets/radhe11/backup/kaggle/working/fingerprint_balanced_test_outputs/fingerprint_test_embeddings.npy",
        "meta_path": "/kaggle/input/datasets/radhe11/backup/kaggle/working/fingerprint_balanced_test_outputs/fingerprint_test_embeddings_meta.csv",
        "out_dim": 1024,
        "binary": False,
    },
    "face": {
        "emb_path": "/kaggle/input/datasets/radhe11/backup/kaggle/working/face_pretrained_embeddings/face_test_embeddings_pretrained.npy",
        "meta_path": "/kaggle/input/datasets/radhe11/backup/kaggle/working/face_pretrained_embeddings/face_test_embeddings_meta_pretrained.csv",
        "out_dim": 256,
        "binary": False,
    },
    "iris": {
        "emb_path": "/kaggle/input/datasets/radhe11/backup/kaggle/working/iris_balanced_test_outputs/iris_test_embeddings.npy",
        "meta_path": "/kaggle/input/datasets/radhe11/backup/kaggle/working/iris_balanced_test_outputs/iris_test_embeddings_meta.csv",
        "out_dim": 1024,
        "binary": False,
    }
}

SAVE_ROOT = f"/kaggle/working/irreversibility_{MODALITY}"
os.makedirs(SAVE_ROOT, exist_ok=True)

TOPK = 5
RANDOM_STATE = 42

# ============================================================
# 2. LOAD
# ============================================================

cfg = CONFIGS[MODALITY]
orig_embeddings = np.load(cfg["emb_path"]).astype(np.float32)
meta = pd.read_csv(cfg["meta_path"])

print("Original embeddings:", orig_embeddings.shape)
print("Recommended config:", {"dim": cfg["out_dim"], "binary": cfg["binary"]})

# ============================================================
# 3. CANCELLABLE TRANSFORM
# ============================================================

def cancellable_transform(emb, key=11, out_dim=512, binary=False):
    rng = np.random.default_rng(seed=key)
    R = rng.standard_normal((emb.shape[1], out_dim))
    proj = emb @ R
    proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
    if binary:
        return (proj > 0).astype(np.float32)
    return proj.astype(np.float32)

protected_embeddings = cancellable_transform(
    orig_embeddings,
    key=KEY,
    out_dim=cfg["out_dim"],
    binary=cfg["binary"]
)

np.save(os.path.join(SAVE_ROOT, f"{MODALITY}_protected_embeddings.npy"), protected_embeddings)

# normalize original too for fair cosine analysis
orig_norm = orig_embeddings / (np.linalg.norm(orig_embeddings, axis=1, keepdims=True) + 1e-12)
prot_norm = protected_embeddings / (np.linalg.norm(protected_embeddings, axis=1, keepdims=True) + 1e-12)

# ============================================================
# 4. DISTANCE / SIMILARITY CORRELATION
# ============================================================

orig_sim = cosine_similarity(orig_norm)
prot_sim = cosine_similarity(prot_norm)

# upper triangular only, excluding diagonal
iu = np.triu_indices_from(orig_sim, k=1)
orig_vals = orig_sim[iu]
prot_vals = prot_sim[iu]

pearson_corr = float(np.corrcoef(orig_vals, prot_vals)[0, 1])
spearman_corr = float(pd.Series(orig_vals).corr(pd.Series(prot_vals), method="spearman"))

corr_results = {
    "modality": MODALITY,
    "recommended_dim": cfg["out_dim"],
    "recommended_binary": cfg["binary"],
    "pearson_similarity_correlation": pearson_corr,
    "spearman_similarity_correlation": spearman_corr,
}

print("\n===== DISTANCE / SIMILARITY CORRELATION =====")
print(corr_results)

# scatter sample for plotting
rng = np.random.default_rng(RANDOM_STATE)
sample_size = min(8000, len(orig_vals))
sample_idx = rng.choice(len(orig_vals), size=sample_size, replace=False)
sample_orig = orig_vals[sample_idx]
sample_prot = prot_vals[sample_idx]

plt.figure(figsize=(6.5, 5))
plt.scatter(sample_orig, sample_prot, s=8, alpha=0.5)
plt.xlabel("Original-space cosine similarity")
plt.ylabel("Protected-space cosine similarity")
plt.title(f"{MODALITY.capitalize()}: Similarity Correlation")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, f"{MODALITY}_similarity_correlation_scatter.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 5. NEAREST-NEIGHBOR OVERLAP
# ============================================================

def topk_neighbors(sim_mat, k=5):
    sim = sim_mat.copy()
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)[:, :k]

orig_nn = topk_neighbors(orig_sim, TOPK)
prot_nn = topk_neighbors(prot_sim, TOPK)

overlaps = []
for i in range(orig_nn.shape[0]):
    a = set(orig_nn[i].tolist())
    b = set(prot_nn[i].tolist())
    overlaps.append(len(a & b) / TOPK)

mean_nn_overlap = float(np.mean(overlaps))
std_nn_overlap = float(np.std(overlaps))

nn_results = {
    "topk": TOPK,
    "mean_neighbor_overlap": mean_nn_overlap,
    "std_neighbor_overlap": std_nn_overlap
}

print("\n===== NEAREST-NEIGHBOR OVERLAP =====")
print(nn_results)

# histogram
plt.figure(figsize=(6.5, 4.8))
plt.hist(overlaps, bins=np.arange(-0.1, 1.11, 0.1), density=False)
plt.xlabel(f"Top-{TOPK} Neighbor Overlap")
plt.ylabel("Count")
plt.title(f"{MODALITY.capitalize()}: Original vs Protected NN Overlap")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, f"{MODALITY}_nn_overlap_hist.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 6. LINEAR RECONSTRUCTION ATTACK BASELINE
# protected -> original
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    prot_norm, orig_norm, test_size=0.30, random_state=RANDOM_STATE
)

# Ridge regression baseline
reconstructor = Ridge(alpha=1.0, random_state=RANDOM_STATE)
reconstructor.fit(X_train, y_train)
recon_test = reconstructor.predict(X_test).astype(np.float32)

# normalize reconstructions before cosine comparison
recon_test_norm = recon_test / (np.linalg.norm(recon_test, axis=1, keepdims=True) + 1e-12)
y_test_norm = y_test / (np.linalg.norm(y_test, axis=1, keepdims=True) + 1e-12)

recon_mse = float(mean_squared_error(y_test_norm, recon_test_norm))
recon_cosines = np.sum(recon_test_norm * y_test_norm, axis=1)
mean_recon_cos = float(np.mean(recon_cosines))
std_recon_cos = float(np.std(recon_cosines))

recon_results = {
    "reconstruction_mse": recon_mse,
    "mean_reconstruction_cosine": mean_recon_cos,
    "std_reconstruction_cosine": std_recon_cos
}

print("\n===== LINEAR RECONSTRUCTION BASELINE =====")
print(recon_results)

plt.figure(figsize=(6.5, 4.8))
plt.hist(recon_cosines, bins=40)
plt.xlabel("Cosine similarity: reconstructed vs original")
plt.ylabel("Count")
plt.title(f"{MODALITY.capitalize()}: Reconstruction Attack Baseline")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, f"{MODALITY}_reconstruction_cosine_hist.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 7. SAVE SUMMARY
# ============================================================

summary = {}
summary.update(corr_results)
summary.update(nn_results)
summary.update(recon_results)

with open(os.path.join(SAVE_ROOT, f"{MODALITY}_irreversibility_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

pd.DataFrame([summary]).to_csv(os.path.join(SAVE_ROOT, f"{MODALITY}_irreversibility_summary.csv"), index=False)

print("\n===== FINAL IRREVERSIBILITY SUMMARY =====")
print(summary)
print("\nSaved in:", SAVE_ROOT)

# ================= NOTEBOOK CELL 111 =================
# ============================================================
# BINARY VS NON-BINARY IRREVERSIBILITY COMPARISON
# At recommended protected dimensions
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    accuracy_score, f1_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ============================================================
# 1. CONFIG
# ============================================================

MODALITY = "fingerprint"   # change to "face" or "iris"
KEYS = [11, 77, 123]
TOPK = 5
RANDOM_STATE = 42

CONFIGS = {
    "fingerprint": {
        "emb_path": "/kaggle/input/datasets/radhe11/backup/kaggle/working/fingerprint_balanced_test_outputs/fingerprint_test_embeddings.npy",
        "meta_path": "/kaggle/input/datasets/radhe11/backup/kaggle/working/fingerprint_balanced_test_outputs/fingerprint_test_embeddings_meta.csv",
        "recommended_dim": 1024,
    },
    "face": {
        "emb_path": "/kaggle/input/datasets/radhe11/backup/kaggle/working/face_pretrained_embeddings/face_test_embeddings_pretrained.npy",
        "meta_path": "/kaggle/input/datasets/radhe11/backup/kaggle/working/face_pretrained_embeddings/face_test_embeddings_meta_pretrained.csv",
        "recommended_dim": 256,
    },
    "iris": {
        "emb_path": "/kaggle/input/datasets/radhe11/backup/kaggle/working/iris_balanced_test_outputs/iris_test_embeddings.npy",
        "meta_path": "/kaggle/input/datasets/radhe11/backup/kaggle/working/iris_balanced_test_outputs/iris_test_embeddings_meta.csv",
        "recommended_dim": 1024,
    }
}

PAIRS_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/common_pairs_balanced/test_pairs_common_balanced.csv"

SAVE_ROOT = f"/kaggle/working/binary_vs_nonbinary_irreversibility_{MODALITY}"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ============================================================
# 2. LOAD
# ============================================================

cfg = CONFIGS[MODALITY]
orig_embeddings = np.load(cfg["emb_path"]).astype(np.float32)
meta = pd.read_csv(cfg["meta_path"])
pairs = pd.read_csv(PAIRS_PATH)

print("Original embeddings:", orig_embeddings.shape)
print("Recommended dimension:", cfg["recommended_dim"])

# ============================================================
# 3. HELPERS
# ============================================================

def cancellable_transform(embeddings, key=11, out_dim=512, binary=False):
    rng = np.random.default_rng(seed=key)
    R = rng.standard_normal((embeddings.shape[1], out_dim))
    proj = embeddings @ R
    proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
    if binary:
        return (proj > 0).astype(np.float32)
    return proj.astype(np.float32)

def build_index_map(meta):
    meta = meta.copy()
    meta["global_index"] = np.arange(len(meta))
    meta = meta.sort_values(["subject", "image_path"]).reset_index(drop=True)
    meta["local_idx"] = meta.groupby("subject").cumcount()
    return {(int(r.subject), int(r.local_idx)): int(r.global_index) for _, r in meta.iterrows()}

def normalize_rows(x):
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def cosine_score(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def evaluate_pairing(probe_templates, gallery_templates, pairs_df, index_map):
    scores = []
    for _, row in pairs_df.iterrows():
        g1 = index_map[(int(row["subject1"]), int(row["idx1"]))]
        g2 = index_map[(int(row["subject2"]), int(row["idx2"]))]
        scores.append(cosine_score(probe_templates[g1], gallery_templates[g2]))

    y_true = pairs_df["label"].values.astype(int)
    y_score = np.array(scores, dtype=np.float32)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(fpr[eer_idx])
    thr = float(thresholds[eer_idx])
    auc = float(roc_auc_score(y_true, y_score))

    if not np.isfinite(thr):
        thr = 0.0

    y_pred = (y_score >= thr).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))

    return {
        "eer": eer,
        "auc": auc,
        "accuracy": acc,
        "f1": f1,
        "threshold": thr,
        "scores": y_score
    }

def nearest_neighbor_overlap(orig_norm, prot_norm, k=5):
    orig_sim = cosine_similarity(orig_norm)
    prot_sim = cosine_similarity(prot_norm)

    np.fill_diagonal(orig_sim, -np.inf)
    np.fill_diagonal(prot_sim, -np.inf)

    orig_nn = np.argsort(-orig_sim, axis=1)[:, :k]
    prot_nn = np.argsort(-prot_sim, axis=1)[:, :k]

    overlaps = []
    for i in range(orig_nn.shape[0]):
        a = set(orig_nn[i].tolist())
        b = set(prot_nn[i].tolist())
        overlaps.append(len(a & b) / k)

    return float(np.mean(overlaps)), float(np.std(overlaps))

def irreversibility_analysis(orig_emb, prot_emb):
    orig_norm = normalize_rows(orig_emb)
    prot_norm = normalize_rows(prot_emb)

    # similarity correlation
    orig_sim = cosine_similarity(orig_norm)
    prot_sim = cosine_similarity(prot_norm)
    iu = np.triu_indices_from(orig_sim, k=1)
    orig_vals = orig_sim[iu]
    prot_vals = prot_sim[iu]

    pearson_corr = float(np.corrcoef(orig_vals, prot_vals)[0, 1])
    spearman_corr = float(pd.Series(orig_vals).corr(pd.Series(prot_vals), method="spearman"))

    # NN overlap
    mean_nn_overlap, std_nn_overlap = nearest_neighbor_overlap(orig_norm, prot_norm, k=TOPK)

    # reconstruction baseline
    X_train, X_test, y_train, y_test = train_test_split(
        prot_norm, orig_norm, test_size=0.30, random_state=RANDOM_STATE
    )
    reconstructor = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    reconstructor.fit(X_train, y_train)
    recon_test = reconstructor.predict(X_test).astype(np.float32)

    recon_test_norm = normalize_rows(recon_test)
    y_test_norm = normalize_rows(y_test)

    recon_mse = float(mean_squared_error(y_test_norm, recon_test_norm))
    recon_cos = np.sum(recon_test_norm * y_test_norm, axis=1)
    mean_recon_cos = float(np.mean(recon_cos))
    std_recon_cos = float(np.std(recon_cos))

    return {
        "pearson_similarity_correlation": pearson_corr,
        "spearman_similarity_correlation": spearman_corr,
        "mean_neighbor_overlap": mean_nn_overlap,
        "std_neighbor_overlap": std_nn_overlap,
        "reconstruction_mse": recon_mse,
        "mean_reconstruction_cosine": mean_recon_cos,
        "std_reconstruction_cosine": std_recon_cos,
    }

# ============================================================
# 4. MAIN COMPARISON
# ============================================================

index_map = build_index_map(meta)

rows = []

for binary in [False, True]:
    label_name = "Non-binary" if not binary else "Binary"
    print(f"\nRunning {MODALITY} | {label_name}")

    # templates for multiple keys
    templates = {}
    for key in KEYS:
        templates[key] = cancellable_transform(
            orig_embeddings,
            key=key,
            out_dim=cfg["recommended_dim"],
            binary=binary
        )

    # ---------- same-key utility ----------
    same_results = []
    for key in KEYS:
        res = evaluate_pairing(templates[key], templates[key], pairs, index_map)
        same_results.append(res)

    same_eer = float(np.mean([r["eer"] for r in same_results]))
    same_auc = float(np.mean([r["auc"] for r in same_results]))
    same_acc = float(np.mean([r["accuracy"] for r in same_results]))
    same_f1  = float(np.mean([r["f1"] for r in same_results]))

    # ---------- cross-key unlinkability ----------
    cross_pairs = [(11, 77), (11, 123), (77, 123)]
    cross_results = []
    cross_sims = []

    for k1, k2 in cross_pairs:
        res = evaluate_pairing(templates[k1], templates[k2], pairs, index_map)
        cross_results.append(res)

        a = normalize_rows(templates[k1])
        b = normalize_rows(templates[k2])
        sims = np.sum(a * b, axis=1)
        cross_sims.append({
            "mean": float(np.mean(sims)),
            "std": float(np.std(sims))
        })

    cross_eer = float(np.mean([r["eer"] for r in cross_results]))
    cross_auc = float(np.mean([r["auc"] for r in cross_results]))
    cross_acc = float(np.mean([r["accuracy"] for r in cross_results]))
    cross_f1  = float(np.mean([r["f1"] for r in cross_results]))
    mean_cross_sim = float(np.mean([x["mean"] for x in cross_sims]))
    std_cross_sim  = float(np.mean([x["std"] for x in cross_sims]))

    # ---------- irreversibility ----------
    irr = irreversibility_analysis(orig_embeddings, templates[11])

    row = {
        "Modality": MODALITY.capitalize(),
        "Template Type": label_name,
        "Dim": cfg["recommended_dim"],

        "Same-key EER": same_eer,
        "Same-key AUC": same_auc,
        "Same-key Accuracy": same_acc,
        "Same-key F1": same_f1,

        "Cross-key EER": cross_eer,
        "Cross-key AUC": cross_auc,
        "Cross-key Accuracy": cross_acc,
        "Cross-key F1": cross_f1,
        "Mean Cross-key Similarity": mean_cross_sim,
        "Std Cross-key Similarity": std_cross_sim,

        "Pearson Corr": irr["pearson_similarity_correlation"],
        "Spearman Corr": irr["spearman_similarity_correlation"],
        "Mean NN Overlap": irr["mean_neighbor_overlap"],
        "Std NN Overlap": irr["std_neighbor_overlap"],
        "Reconstruction MSE": irr["reconstruction_mse"],
        "Mean Reconstruction Cosine": irr["mean_reconstruction_cosine"],
        "Std Reconstruction Cosine": irr["std_reconstruction_cosine"],
    }
    rows.append(row)

comparison_df = pd.DataFrame(rows)
comparison_csv = os.path.join(SAVE_ROOT, f"{MODALITY}_binary_vs_nonbinary_irreversibility.csv")
comparison_df.to_csv(comparison_csv, index=False)

print("\n===== FINAL COMPARISON TABLE =====")
print(comparison_df)

# ============================================================
# 5. CLEAN COMPARISON CHARTS
# ============================================================

labels = comparison_df["Template Type"].tolist()
x = np.arange(len(labels))

# same-key EER
plt.figure(figsize=(6.5, 4.5))
plt.bar(labels, comparison_df["Same-key EER"])
plt.ylabel("Same-key EER")
plt.title(f"{MODALITY.capitalize()}: Utility Comparison")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, f"{MODALITY}_samekey_eer_binary_vs_nonbinary.png"), dpi=300, bbox_inches="tight")
plt.show()

# cross-key AUC
plt.figure(figsize=(6.5, 4.5))
plt.bar(labels, comparison_df["Cross-key AUC"])
plt.ylabel("Cross-key ROC-AUC")
plt.title(f"{MODALITY.capitalize()}: Unlinkability Comparison")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, f"{MODALITY}_crosskey_auc_binary_vs_nonbinary.png"), dpi=300, bbox_inches="tight")
plt.show()

# irreversibility correlations
plt.figure(figsize=(6.5, 4.5))
w = 0.35
plt.bar(x - w/2, comparison_df["Pearson Corr"], width=w, label="Pearson")
plt.bar(x + w/2, comparison_df["Mean Reconstruction Cosine"], width=w, label="Recon Cosine")
plt.xticks(x, labels)
plt.ylabel("Value")
plt.title(f"{MODALITY.capitalize()}: Irreversibility Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, f"{MODALITY}_irreversibility_binary_vs_nonbinary.png"), dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved in:", SAVE_ROOT)
