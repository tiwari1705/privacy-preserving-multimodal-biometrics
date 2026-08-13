"""Extracted from notebook3e7611700a(1).ipynb.
Cells preserved in original execution order.
Kaggle paths are intentionally preserved; replace them for local execution.
"""

# ================= NOTEBOOK CELL 35 =================
# ============================================================
# FACE EMBEDDING EXTRACTION USING PRETRAINED INCEPTIONRESNETV1
# pretrained = 'vggface2'
# Extract embeddings for train / val / test / all
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from facenet_pytorch import InceptionResnetV1

# =========================
# 1. CONFIG
# =========================
IMG_SIZE = 160
BATCH_SIZE = 64
NUM_WORKERS = 2

DATASET_ROOT = "/kaggle/input/datasets/radhe11/multimodal"
SPLIT_DIR = "/kaggle/working/splits"
SAVE_DIR = "/kaggle/working/face_pretrained_embeddings"

TRAIN_CSV = os.path.join(SPLIT_DIR, "train_subjects.csv")
VAL_CSV   = os.path.join(SPLIT_DIR, "val_subjects.csv")
TEST_CSV  = os.path.join(SPLIT_DIR, "test_subjects.csv")

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# 2. HELPERS
# =========================
def clean_sid(x):
    return str(int(float(x))).zfill(3)

def read_subject_ids(csv_path):
    df = pd.read_csv(csv_path)
    return df.iloc[:, 0].dropna().apply(clean_sid).tolist()

def natural_sort_key(s):
    import re
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', str(s))]

def list_face_images(subject_id, dataset_root):
    sid = clean_sid(subject_id)
    folder = os.path.join(dataset_root, sid, "face")
    if not os.path.isdir(folder):
        return []

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    ]
    files = sorted(files, key=natural_sort_key)
    return files

def load_face_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

def build_image_dataframe(subject_ids, dataset_root, split_name="split"):
    rows = []
    skipped = []

    for sid in tqdm(subject_ids, desc=f"Scanning {split_name} face images"):
        imgs = list_face_images(sid, dataset_root)
        if len(imgs) == 0:
            skipped.append(sid)
            continue

        for idx, img_path in enumerate(imgs):
            rows.append([sid, idx, img_path])

    df = pd.DataFrame(rows, columns=["subject", "img_idx", "image_path"])
    return df, skipped

# =========================
# 3. LOAD SPLITS
# =========================
train_ids = read_subject_ids(TRAIN_CSV)
val_ids   = read_subject_ids(VAL_CSV)
test_ids  = read_subject_ids(TEST_CSV)

train_df, skipped_train = build_image_dataframe(train_ids, DATASET_ROOT, "train")
val_df, skipped_val     = build_image_dataframe(val_ids, DATASET_ROOT, "val")
test_df, skipped_test   = build_image_dataframe(test_ids, DATASET_ROOT, "test")

all_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)

print("\nTrain images:", len(train_df), "| skipped:", len(skipped_train))
print("Val images  :", len(val_df),   "| skipped:", len(skipped_val))
print("Test images :", len(test_df),  "| skipped:", len(skipped_test))
print("All images  :", len(all_df))

# save meta
train_df.to_csv(os.path.join(SAVE_DIR, "train_image_meta.csv"), index=False)
val_df.to_csv(os.path.join(SAVE_DIR, "val_image_meta.csv"), index=False)
test_df.to_csv(os.path.join(SAVE_DIR, "test_image_meta.csv"), index=False)
all_df.to_csv(os.path.join(SAVE_DIR, "all_image_meta.csv"), index=False)

# =========================
# 4. TRANSFORM
# facenet-pytorch generally expects 160x160
# =========================
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# =========================
# 5. DATASET
# =========================
class FaceImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_face_rgb(row["image_path"])
        if self.transform:
            img = self.transform(img)
        return img, row["subject"], int(row["img_idx"]), row["image_path"]

def make_loader(df):
    ds = FaceImageDataset(df, transform=eval_transform)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

train_loader = make_loader(train_df)
val_loader   = make_loader(val_df)
test_loader  = make_loader(test_df)
all_loader   = make_loader(all_df)

# =========================
# 6. LOAD PRETRAINED MODEL
# pretrained='vggface2'
# =========================
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("\nLoaded InceptionResnetV1 pretrained on VGGFace2")

# =========================
# 7. EXTRACT EMBEDDINGS
# =========================
@torch.no_grad()
def extract_embeddings(model, loader, split_name="split"):
    model.eval()

    all_embs = []
    all_subjects = []
    all_indices = []
    all_paths = []

    pbar = tqdm(loader, total=len(loader), desc=f"Extracting {split_name} face embeddings")
    for imgs, subjects, img_indices, paths in pbar:
        imgs = imgs.to(device, non_blocking=True)

        embs = model(imgs)   # shape [B, 512]
        embs = embs.detach().cpu().numpy()

        all_embs.append(embs)
        all_subjects.extend(list(subjects))
        all_indices.extend([int(x) for x in img_indices])
        all_paths.extend(list(paths))

    all_embs = np.concatenate(all_embs, axis=0)

    meta_df = pd.DataFrame({
        "subject": all_subjects,
        "img_idx": all_indices,
        "image_path": all_paths
    })

    return all_embs, meta_df

train_embs, train_meta = extract_embeddings(model, train_loader, "train")
val_embs, val_meta     = extract_embeddings(model, val_loader, "val")
test_embs, test_meta   = extract_embeddings(model, test_loader, "test")
all_embs, all_meta     = extract_embeddings(model, all_loader, "all")

print("\nEmbedding shapes:")
print("Train:", train_embs.shape)
print("Val  :", val_embs.shape)
print("Test :", test_embs.shape)
print("All  :", all_embs.shape)

# =========================
# 8. SAVE EMBEDDINGS
# =========================
np.save(os.path.join(SAVE_DIR, "face_train_embeddings_pretrained.npy"), train_embs)
np.save(os.path.join(SAVE_DIR, "face_val_embeddings_pretrained.npy"), val_embs)
np.save(os.path.join(SAVE_DIR, "face_test_embeddings_pretrained.npy"), test_embs)
np.save(os.path.join(SAVE_DIR, "face_all_embeddings_pretrained.npy"), all_embs)

train_meta.to_csv(os.path.join(SAVE_DIR, "face_train_embeddings_meta_pretrained.csv"), index=False)
val_meta.to_csv(os.path.join(SAVE_DIR, "face_val_embeddings_meta_pretrained.csv"), index=False)
test_meta.to_csv(os.path.join(SAVE_DIR, "face_test_embeddings_meta_pretrained.csv"), index=False)
all_meta.to_csv(os.path.join(SAVE_DIR, "face_all_embeddings_meta_pretrained.csv"), index=False)

print("\n✅ Saved files:")
for f in sorted(os.listdir(SAVE_DIR)):
    print(os.path.join(SAVE_DIR, f))

print("\n✅ DONE")

# ================= NOTEBOOK CELL 36 =================
# ============================================================
# FACE PRETRAINED EMBEDDINGS -> BALANCED TEST SCORING
# + METRICS + ROC + CONFUSION MATRIX
# ============================================================

import os
import json
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
# 1. PATHS
# =========================
EMB_DIR = "/kaggle/working/face_pretrained_embeddings"
PAIR_DIR = "/kaggle/working/common_pairs_balanced"
SAVE_DIR = "/kaggle/working/face_pretrained_test_outputs"

os.makedirs(SAVE_DIR, exist_ok=True)

TEST_EMB_PATH = os.path.join(EMB_DIR, "face_test_embeddings_pretrained.npy")
TEST_META_PATH = os.path.join(EMB_DIR, "face_test_embeddings_meta_pretrained.csv")
PAIR_CSV = os.path.join(PAIR_DIR, "test_pairs_common_balanced.csv")

# =========================
# 2. HELPERS
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
# 3. LOAD
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
# 4. BUILD EMBEDDING MAP
# =========================
emb_map = {}
for i, row in test_meta.iterrows():
    emb_map[(row["subject"], int(row["img_idx"]))] = test_embs[i]

print("Embedding map size:", len(emb_map))

# =========================
# 5. COMPUTE SCORES
# =========================
scores = []
labels = []
valid_rows = []

for _, row in pair_df.iterrows():
    pair_id = int(row["pair_id"])
    s1 = row["subject1"]
    s2 = row["subject2"]
    idx1 = int(row["idx1"])
    idx2 = int(row["idx2"])
    label = int(row["label"])

    k1 = (s1, idx1)
    k2 = (s2, idx2)

    if k1 not in emb_map or k2 not in emb_map:
        continue

    e1 = emb_map[k1]
    e2 = emb_map[k2]

    score = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12))

    scores.append(score)
    labels.append(label)
    valid_rows.append([pair_id, s1, s2, idx1, idx2, label, score])

score_df = pd.DataFrame(
    valid_rows,
    columns=["pair_id", "subject1", "subject2", "idx1", "idx2", "label", "score"]
)

score_df.to_csv(os.path.join(SAVE_DIR, "face_test_scores_pretrained_balanced.csv"), index=False)

scores = np.array(scores)
labels = np.array(labels)

print("Pairs requested :", len(pair_df))
print("Pairs evaluated :", len(score_df))

# =========================
# 6. EVALUATION
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
    "tar_at_far_0_1_percent": float(tar01),
    "tp": int(TP),
    "tn": int(TN),
    "fp": int(FP),
    "fn": int(FN),
    "num_pairs_evaluated": int(len(score_df)),
    "embedding_dim": int(test_embs.shape[1])
}

with open(os.path.join(SAVE_DIR, "face_test_metrics_pretrained_balanced.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("\n🔥 FACE PRETRAINED BALANCED TEST METRICS")
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
# 7. ROC CURVE
# =========================
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Face Pretrained Balanced ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Face Pretrained Balanced ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "face_test_roc_curve_pretrained_balanced.png"), dpi=200)
plt.close()

# =========================
# 8. SCORE DISTRIBUTION
# =========================
genuine_scores = scores[labels == 1]
impostor_scores = scores[labels == 0]

plt.figure(figsize=(7,5))
plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
plt.axvline(threshold, linestyle="--", label=f"Thr@EER={threshold:.4f}")
plt.xlabel("Cosine Similarity Score")
plt.ylabel("Density")
plt.title("Face Pretrained Balanced Score Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
plt.axvline(threshold, linestyle="--", label=f"Thr@EER={threshold:.4f}")
plt.xlabel("Cosine Similarity Score")
plt.ylabel("Density")
plt.title("Face Pretrained Balanced Score Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "face_test_score_distribution_pretrained_balanced.png"), dpi=200)
plt.close()

# =========================
# 9. CONFUSION MATRIX
# =========================
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation="nearest")
plt.title("Face Pretrained Balanced Confusion Matrix")
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
plt.title("Face Pretrained Balanced Confusion Matrix")
plt.colorbar()
plt.xticks(ticks, ["Impostor", "Genuine"])
plt.yticks(ticks, ["Impostor", "Genuine"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "face_test_confusion_matrix_pretrained_balanced.png"), dpi=200)
plt.close()

roc_df = pd.DataFrame({
    "fpr": fpr,
    "tpr": tpr,
    "threshold": thresholds
})
roc_df.to_csv(os.path.join(SAVE_DIR, "face_test_roc_points_pretrained_balanced.csv"), index=False)

print("\nConfusion Matrix:")
print(cm)

print("\n✅ DONE")
