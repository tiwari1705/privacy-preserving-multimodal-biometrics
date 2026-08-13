"""Extracted from notebook3e7611700a(1).ipynb.
Cells preserved in original execution order.
Kaggle paths are intentionally preserved; replace them for local execution.
"""

# ================= NOTEBOOK CELL 39 =================
# ============================================================
# IRIS PIPELINE (HIGH-QUALITY BASELINE)
# - Backbone: EfficientNet-B3 pretrained
# - Train on train split
# - Validate using balanced val pairs
# - Save best model by lowest val EER
# - Extract embeddings for train/val/test/all
# - Compute balanced test scores
# - Evaluate metrics
# - Show ROC, score distribution, confusion matrix
# ============================================================

import os
import gc
import cv2
import json
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

warnings.filterwarnings("ignore")

# =========================
# 1. CONFIG
# =========================
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 24
NUM_WORKERS = 2
EMBED_DIM = 512
EPOCHS = 20

HEAD_ONLY_EPOCHS = 3
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4

DATASET_ROOT = "/kaggle/input/datasets/radhe11/multimodal"
SPLIT_DIR = "/kaggle/working/splits"
PAIR_DIR = "/kaggle/working/common_pairs_balanced"

TRAIN_CSV = os.path.join(SPLIT_DIR, "train_subjects.csv")
VAL_CSV   = os.path.join(SPLIT_DIR, "val_subjects.csv")
TEST_CSV  = os.path.join(SPLIT_DIR, "test_subjects.csv")

VAL_PAIR_CSV  = os.path.join(PAIR_DIR, "val_pairs_common_balanced.csv")
TEST_PAIR_CSV = os.path.join(PAIR_DIR, "test_pairs_common_balanced.csv")

TRAIN_SAVE_DIR = "/kaggle/working/iris_training_balanced"
FINAL_SAVE_DIR = "/kaggle/working/iris_balanced_test_outputs"

os.makedirs(TRAIN_SAVE_DIR, exist_ok=True)
os.makedirs(FINAL_SAVE_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(TRAIN_SAVE_DIR, "best_iris_model.pth")
LAST_MODEL_PATH = os.path.join(TRAIN_SAVE_DIR, "last_iris_model.pth")
HISTORY_PATH = os.path.join(TRAIN_SAVE_DIR, "training_history.csv")
CLASS_MAP_PATH = os.path.join(TRAIN_SAVE_DIR, "train_class_mapping.csv")
CONFIG_PATH = os.path.join(TRAIN_SAVE_DIR, "train_config.json")

# =========================
# 2. REPRODUCIBILITY
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# 3. HELPERS
# =========================
def clean_sid(x):
    return str(int(float(x))).zfill(3)

def read_subject_ids(csv_path):
    df = pd.read_csv(csv_path)
    return df.iloc[:, 0].dropna().apply(clean_sid).tolist()

def natural_sort_key(s):
    import re
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', str(s))]

def list_iris_images(subject_id, dataset_root):
    sid = clean_sid(subject_id)
    folder = os.path.join(dataset_root, sid, "iris")
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

def load_iris_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

def build_train_dataframe(subject_ids, dataset_root):
    rows = []
    skipped = []

    for sid in tqdm(subject_ids, desc="Scanning train iris images"):
        imgs = list_iris_images(sid, dataset_root)
        if len(imgs) == 0:
            skipped.append(sid)
            continue

        for img_path in imgs:
            rows.append([sid, img_path])

    df = pd.DataFrame(rows, columns=["subject", "image_path"])
    return df, skipped

def build_image_dataframe(subject_ids, dataset_root, split_name="split"):
    rows = []
    skipped = []

    for sid in tqdm(subject_ids, desc=f"Scanning {split_name} iris images"):
        imgs = list_iris_images(sid, dataset_root)

        if len(imgs) == 0:
            skipped.append(sid)
            continue

        for idx, img_path in enumerate(imgs):
            rows.append([sid, idx, img_path])

    df = pd.DataFrame(rows, columns=["subject", "img_idx", "image_path"])
    return df, skipped

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
# 4. LOAD SPLITS / PAIRS
# =========================
train_ids = read_subject_ids(TRAIN_CSV)
val_ids = read_subject_ids(VAL_CSV)
test_ids = read_subject_ids(TEST_CSV)

val_pairs_df = pd.read_csv(VAL_PAIR_CSV)
test_pairs_df = pd.read_csv(TEST_PAIR_CSV)

val_pairs_df["subject1"] = val_pairs_df["subject1"].apply(clean_sid)
val_pairs_df["subject2"] = val_pairs_df["subject2"].apply(clean_sid)

test_pairs_df["subject1"] = test_pairs_df["subject1"].apply(clean_sid)
test_pairs_df["subject2"] = test_pairs_df["subject2"].apply(clean_sid)

print("Train IDs:", len(train_ids))
print("Val IDs:", len(val_ids))
print("Test IDs:", len(test_ids))
print("Val pairs :", val_pairs_df.shape)
print("Test pairs:", test_pairs_df.shape)

train_df, skipped_train = build_train_dataframe(train_ids, DATASET_ROOT)

print("Train images:", len(train_df))
print("Unique train subjects found:", train_df["subject"].nunique() if len(train_df) else 0)
print("Skipped train subjects with no iris images:", len(skipped_train))
print(train_df.head())

if len(train_df) == 0:
    raise ValueError("No training iris images found.")

# labels
le = LabelEncoder()
train_df["label"] = le.fit_transform(train_df["subject"].astype(str))
num_classes = train_df["label"].nunique()
print("Num train classes:", num_classes)

class_map_df = pd.DataFrame({
    "subject": le.classes_,
    "label": np.arange(len(le.classes_))
})
class_map_df.to_csv(CLASS_MAP_PATH, index=False)

# =========================
# 5. TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.08, contrast=0.08),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# =========================
# 6. DATASET
# =========================
class IrisTrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_iris_rgb(row["image_path"])
        if self.transform:
            img = self.transform(img)
        label = int(row["label"])
        return img, label

class IrisImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_iris_rgb(row["image_path"])
        if self.transform:
            img = self.transform(img)
        return img, row["subject"], int(row["img_idx"]), row["image_path"]

train_dataset = IrisTrainDataset(train_df, transform=train_transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# =========================
# 7. MODEL
# =========================
class IrisEmbeddingNet(nn.Module):
    def __init__(self, num_classes, embed_dim=512):
        super().__init__()
        backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_embedding=False):
        feat = self.backbone(x)
        emb = self.embedding(feat)
        emb = F.normalize(emb, p=2, dim=1)

        if return_embedding:
            return emb

        logits = self.classifier(emb)
        return logits, emb

model = IrisEmbeddingNet(num_classes=num_classes, embed_dim=EMBED_DIM).to(device)

for p in model.backbone.parameters():
    p.requires_grad = False

print("Model loaded")

# =========================
# 8. LOSS / OPTIMIZER / SCHEDULER / AMP
# =========================
criterion = nn.CrossEntropyLoss()
val_bce = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_HEAD,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 9. TRAIN FUNCTION
# =========================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()

    running_loss = 0.0
    all_true = []
    all_pred = []

    pbar = tqdm(loader, total=len(loader), desc="Train", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits, _ = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)

        preds = logits.argmax(dim=1)
        all_true.extend(labels.detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_true, all_pred)
    return epoch_loss, epoch_acc

# =========================
# 10. VALIDATION ON COMMON PAIRS
# =========================
@torch.no_grad()
def extract_subject_embeddings_for_val(model, subject_ids, dataset_root, transform, device):
    model.eval()
    emb_map = {}

    for sid in tqdm(subject_ids, desc="Extracting val iris embeddings", leave=False):
        img_paths = list_iris_images(sid, dataset_root)
        for idx, path in enumerate(img_paths):
            img = load_iris_rgb(path)
            img = transform(img).unsqueeze(0).to(device)

            emb = model(img, return_embedding=True)
            emb = emb.squeeze(0).detach().cpu().numpy()
            emb_map[(sid, idx)] = emb

    return emb_map

@torch.no_grad()
def evaluate_on_common_pairs(model, pair_df, subject_ids, dataset_root, transform, device):
    emb_map = extract_subject_embeddings_for_val(model, subject_ids, dataset_root, transform, device)

    scores = []
    labels = []
    logits_for_bce = []
    valid_pair_ids = []

    for _, row in pair_df.iterrows():
        s1 = clean_sid(row["subject1"])
        s2 = clean_sid(row["subject2"])
        idx1 = int(row["idx1"])
        idx2 = int(row["idx2"])
        label = int(row["label"])
        pair_id = int(row["pair_id"])

        k1 = (s1, idx1)
        k2 = (s2, idx2)

        if k1 not in emb_map or k2 not in emb_map:
            continue

        e1 = emb_map[k1]
        e2 = emb_map[k2]

        score = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12))

        score_clamped = np.clip(score, -0.999999, 0.999999)
        prob = (score_clamped + 1.0) / 2.0
        logit = np.log(prob / (1.0 - prob))

        scores.append(score)
        labels.append(label)
        logits_for_bce.append(logit)
        valid_pair_ids.append(pair_id)

    scores = np.array(scores)
    labels = np.array(labels)
    logits_for_bce = np.array(logits_for_bce, dtype=np.float32)

    if len(scores) == 0:
        raise ValueError("No validation pairs could be evaluated for iris.")

    eer, threshold, _, _, _ = compute_eer(scores, labels)
    preds = (scores >= threshold).astype(int)
    acc = accuracy_score(labels, preds)

    val_loss = val_bce(
        torch.tensor(logits_for_bce),
        torch.tensor(labels.astype(np.float32))
    ).item()

    metrics = {
        "num_pairs_used": int(len(scores)),
        "val_loss": float(val_loss),
        "eer": float(eer),
        "threshold": float(threshold),
        "accuracy": float(acc)
    }
    return metrics, scores, labels, valid_pair_ids

# =========================
# 11. TRAIN LOOP
# =========================
history = []
best_eer = 999.0
best_epoch = -1

print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")

    if epoch == HEAD_ONLY_EPOCHS + 1:
        print("🔥 Backbone Unfrozen")
        for p in model.backbone.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
            {"params": model.embedding.parameters(), "lr": LR_HEAD},
            {"params": model.classifier.parameters(), "lr": LR_HEAD},
        ], weight_decay=WEIGHT_DECAY)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )

    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, scaler, device
    )

    val_metrics, val_scores, val_labels, val_pair_ids = evaluate_on_common_pairs(
        model=model,
        pair_df=val_pairs_df,
        subject_ids=val_ids,
        dataset_root=DATASET_ROOT,
        transform=eval_transform,
        device=device
    )

    scheduler.step(val_metrics["eer"])

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_metrics["val_loss"],
        "val_eer": val_metrics["eer"],
        "val_threshold": val_metrics["threshold"],
        "val_acc": val_metrics["accuracy"],
        "val_pairs_used": val_metrics["num_pairs_used"],
        "lr_group0": optimizer.param_groups[0]["lr"]
    }
    history.append(row)

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.4f}")
    print(f"Val Loss   : {val_metrics['val_loss']:.4f}")
    print(f"Val EER    : {val_metrics['eer']*100:.2f}%")
    print(f"Val Thr    : {val_metrics['threshold']:.4f}")
    print(f"Val Acc    : {val_metrics['accuracy']:.4f}")
    print(f"Pairs Used : {val_metrics['num_pairs_used']}")

    if val_metrics["eer"] < best_eer:
        best_eer = val_metrics["eer"]
        best_epoch = epoch

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_eer": best_eer,
            "embed_dim": EMBED_DIM,
            "num_classes": num_classes,
            "img_size": IMG_SIZE
        }, BEST_MODEL_PATH)

        val_score_df = pd.DataFrame({
            "pair_id": val_pair_ids,
            "label": val_labels,
            "score": val_scores
        })
        val_score_df.to_csv(os.path.join(TRAIN_SAVE_DIR, "best_val_scores.csv"), index=False)

        print("✅ BEST MODEL SAVED")

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# save last model
torch.save({
    "epoch": EPOCHS,
    "model_state_dict": model.state_dict(),
    "best_eer": best_eer,
    "embed_dim": EMBED_DIM,
    "num_classes": num_classes,
    "img_size": IMG_SIZE
}, LAST_MODEL_PATH)

config = {
    "seed": SEED,
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "embed_dim": EMBED_DIM,
    "epochs": EPOCHS,
    "head_only_epochs": HEAD_ONLY_EPOCHS,
    "lr_head": LR_HEAD,
    "lr_backbone": LR_BACKBONE,
    "weight_decay": WEIGHT_DECAY,
    "dataset_root": DATASET_ROOT,
    "train_csv": TRAIN_CSV,
    "val_csv": VAL_CSV,
    "val_pair_csv": VAL_PAIR_CSV,
    "best_epoch": best_epoch,
    "best_val_eer": best_eer
}
with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=4)

print("\n==============================")
print("✅ IRIS TRAINING COMPLETE")
print("==============================")
print(f"Best epoch    : {best_epoch}")
print(f"Best val EER  : {best_eer*100:.2f}%")
print(f"Best model    : {BEST_MODEL_PATH}")

# =========================
# 12. BUILD IMAGE DFS FOR EMBEDDING EXTRACTION
# =========================
train_img_df, _ = build_image_dataframe(train_ids, DATASET_ROOT, "train")
val_img_df, _   = build_image_dataframe(val_ids, DATASET_ROOT, "val")
test_img_df, _  = build_image_dataframe(test_ids, DATASET_ROOT, "test")
all_img_df = pd.concat([train_img_df, val_img_df, test_img_df], axis=0).reset_index(drop=True)

train_img_df.to_csv(os.path.join(FINAL_SAVE_DIR, "train_image_meta.csv"), index=False)
val_img_df.to_csv(os.path.join(FINAL_SAVE_DIR, "val_image_meta.csv"), index=False)
test_img_df.to_csv(os.path.join(FINAL_SAVE_DIR, "test_image_meta.csv"), index=False)
all_img_df.to_csv(os.path.join(FINAL_SAVE_DIR, "all_image_meta.csv"), index=False)

def make_loader(df):
    ds = IrisImageDataset(df, transform=eval_transform)
    return DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

train_img_loader = make_loader(train_img_df)
val_img_loader   = make_loader(val_img_df)
test_img_loader  = make_loader(test_img_df)
all_img_loader   = make_loader(all_img_df)

# =========================
# 13. LOAD BEST MODEL FOR EXTRACTION
# =========================
ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# =========================
# 14. EXTRACT EMBEDDINGS
# =========================
@torch.no_grad()
def extract_embeddings(model, loader, split_name="split"):
    model.eval()

    all_embs = []
    all_subjects = []
    all_indices = []
    all_paths = []

    pbar = tqdm(loader, total=len(loader), desc=f"Extracting {split_name} iris embeddings")
    for imgs, subjects, img_indices, paths in pbar:
        imgs = imgs.to(device, non_blocking=True)
        embs = model(imgs, return_embedding=True).detach().cpu().numpy()

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

train_embs, train_meta = extract_embeddings(model, train_img_loader, "train")
val_embs, val_meta     = extract_embeddings(model, val_img_loader, "val")
test_embs, test_meta   = extract_embeddings(model, test_img_loader, "test")
all_embs, all_meta     = extract_embeddings(model, all_img_loader, "all")

print("\nEmbedding shapes:")
print("Train:", train_embs.shape)
print("Val  :", val_embs.shape)
print("Test :", test_embs.shape)
print("All  :", all_embs.shape)

np.save(os.path.join(FINAL_SAVE_DIR, "iris_train_embeddings.npy"), train_embs)
np.save(os.path.join(FINAL_SAVE_DIR, "iris_val_embeddings.npy"), val_embs)
np.save(os.path.join(FINAL_SAVE_DIR, "iris_test_embeddings.npy"), test_embs)
np.save(os.path.join(FINAL_SAVE_DIR, "iris_all_embeddings.npy"), all_embs)

train_meta.to_csv(os.path.join(FINAL_SAVE_DIR, "iris_train_embeddings_meta.csv"), index=False)
val_meta.to_csv(os.path.join(FINAL_SAVE_DIR, "iris_val_embeddings_meta.csv"), index=False)
test_meta.to_csv(os.path.join(FINAL_SAVE_DIR, "iris_test_embeddings_meta.csv"), index=False)
all_meta.to_csv(os.path.join(FINAL_SAVE_DIR, "iris_all_embeddings_meta.csv"), index=False)

# =========================
# 15. BUILD TEST EMBEDDING MAP
# =========================
test_emb_map = {}
for i, row in test_meta.iterrows():
    test_emb_map[(clean_sid(row["subject"]), int(row["img_idx"]))] = test_embs[i]

print("\nTest embedding map size:", len(test_emb_map))

# =========================
# 16. COMPUTE BALANCED TEST SCORES
# =========================
scores = []
labels = []
valid_rows = []

for _, row in tqdm(test_pairs_df.iterrows(), total=len(test_pairs_df), desc="Computing balanced iris test scores"):
    pair_id = int(row["pair_id"])
    s1 = clean_sid(row["subject1"])
    s2 = clean_sid(row["subject2"])
    idx1 = int(row["idx1"])
    idx2 = int(row["idx2"])
    label = int(row["label"])

    k1 = (s1, idx1)
    k2 = (s2, idx2)

    if k1 not in test_emb_map or k2 not in test_emb_map:
        continue

    e1 = test_emb_map[k1]
    e2 = test_emb_map[k2]

    score = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12))
    scores.append(score)
    labels.append(label)
    valid_rows.append([pair_id, s1, s2, idx1, idx2, label, score])

score_df = pd.DataFrame(
    valid_rows,
    columns=["pair_id", "subject1", "subject2", "idx1", "idx2", "label", "score"]
)
score_df.to_csv(os.path.join(FINAL_SAVE_DIR, "iris_test_scores_balanced.csv"), index=False)

scores = np.array(scores)
labels = np.array(labels)

print("\nPairs requested :", len(test_pairs_df))
print("Pairs evaluated :", len(score_df))

if len(scores) == 0:
    raise ValueError("No valid iris test pairs could be scored.")

# =========================
# 17. EVALUATE METRICS
# =========================
eer, threshold, fpr, tpr, thresholds = compute_eer(scores, labels)
roc_auc = auc(fpr, tpr)

preds = (scores >= threshold).astype(int)

acc  = accuracy_score(labels, preds)
prec = precision_score(labels, preds, zero_division=0)
rec  = recall_score(labels, preds, zero_division=0)
f1   = f1_score(labels, preds, zero_division=0)

cm = confusion_matrix(labels, preds)
TN, FP, FN, TP = cm.ravel()
far = FP / (FP + TN)
frr = FN / (FN + TP)

tar1, actual_far1, thr_far1 = tar_at_far(labels, scores, target_far=0.01)
tar01, actual_far01, thr_far01 = tar_at_far(labels, scores, target_far=0.001)

metrics = {
    "num_test_pairs_requested": int(len(test_pairs_df)),
    "num_test_pairs_evaluated": int(len(score_df)),
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
    "embedding_dim": int(EMBED_DIM)
}

with open(os.path.join(FINAL_SAVE_DIR, "iris_test_metrics_balanced.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("\n🔥 IRIS BALANCED TEST METRICS")
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
# 18. SHOW CURVES
# =========================
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Iris Balanced Test ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Iris Balanced Test ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FINAL_SAVE_DIR, "iris_test_roc_curve_balanced.png"), dpi=200)
plt.close()

genuine_scores = scores[labels == 1]
impostor_scores = scores[labels == 0]

plt.figure(figsize=(7, 5))
plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
plt.axvline(threshold, linestyle="--", label=f"Thr@EER={threshold:.4f}")
plt.xlabel("Cosine Similarity Score")
plt.ylabel("Density")
plt.title("Iris Balanced Test Score Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(genuine_scores, bins=50, alpha=0.6, density=True, label="Genuine")
plt.hist(impostor_scores, bins=50, alpha=0.6, density=True, label="Impostor")
plt.axvline(threshold, linestyle="--", label=f"Thr@EER={threshold:.4f}")
plt.xlabel("Cosine Similarity Score")
plt.ylabel("Density")
plt.title("Iris Balanced Test Score Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FINAL_SAVE_DIR, "iris_test_score_distribution_balanced.png"), dpi=200)
plt.close()

plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Iris Balanced Test Confusion Matrix")
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
plt.title("Iris Balanced Test Confusion Matrix")
plt.colorbar()
plt.xticks(ticks, ["Impostor", "Genuine"])
plt.yticks(ticks, ["Impostor", "Genuine"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(FINAL_SAVE_DIR, "iris_test_confusion_matrix_balanced.png"), dpi=200)
plt.close()

roc_df = pd.DataFrame({
    "fpr": fpr,
    "tpr": tpr,
    "threshold": thresholds
})
roc_df.to_csv(os.path.join(FINAL_SAVE_DIR, "iris_test_roc_points_balanced.csv"), index=False)

with open(os.path.join(FINAL_SAVE_DIR, "iris_test_summary_balanced.txt"), "w") as f:
    f.write("IRIS BALANCED TEST SUMMARY\n")
    f.write("==========================\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")

print("\nConfusion Matrix:")
print(cm)

print("\n✅ SAVED FILES:")
for fname in sorted(os.listdir(FINAL_SAVE_DIR)):
    print(os.path.join(FINAL_SAVE_DIR, fname))

print("\n✅ DONE")
