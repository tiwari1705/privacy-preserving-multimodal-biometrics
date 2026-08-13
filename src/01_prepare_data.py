"""Extracted from notebook3e7611700a(1).ipynb.
Cells preserved in original execution order.
Kaggle paths are intentionally preserved; replace them for local execution.
"""

# ================= NOTEBOOK CELL 1 =================
import os
import random
import pandas as pd

# ----------------------------
# 🔹 Root directory of your dataset
# ----------------------------
root_dir = "/kaggle/input/datasets/radhe11/multimodal"

# ----------------------------
# 🔹 Get all subject IDs
# ----------------------------
subjects = sorted(os.listdir(root_dir))
print("Total subjects:", len(subjects))

# ----------------------------
# 🔹 Shuffle subjects (reproducible)
# ----------------------------
random.seed(42)
random.shuffle(subjects)

# ----------------------------
# 🔹 Split indices (80-10-10)
# ----------------------------
train_split = int(0.8 * len(subjects))
val_split   = int(0.9 * len(subjects))

train_ids = subjects[:train_split]
val_ids   = subjects[train_split:val_split]
test_ids  = subjects[val_split:]

print(f"Train: {len(train_ids)} subjects")
print(f"Val:   {len(val_ids)} subjects")
print(f"Test:  {len(test_ids)} subjects")

# ----------------------------
# 🔹 Function to validate and collect subject data
# ----------------------------
def collect_subjects(subject_list):
    valid_data = []

    for sid in subject_list:
        subj_path = os.path.join(root_dir, sid)

        face_path = os.path.join(subj_path, "face")
        finger_path = os.path.join(subj_path, "fingerprint")
        iris_path = os.path.join(subj_path, "iris")

        # Check all modalities exist
        if not (os.path.exists(face_path) and os.path.exists(finger_path) and os.path.exists(iris_path)):
            print(f"❌ Missing modality for subject: {sid}")
            continue

        # Get sorted images
        face_imgs = sorted([os.path.join(face_path, f) for f in os.listdir(face_path)])
        finger_imgs = sorted([os.path.join(finger_path, f) for f in os.listdir(finger_path)])
        iris_imgs = sorted([os.path.join(iris_path, f) for f in os.listdir(iris_path)])

        # Skip empty folders
        if len(face_imgs) == 0 or len(finger_imgs) == 0 or len(iris_imgs) == 0:
            print(f"❌ Empty folder in subject: {sid}")
            continue

        valid_data.append({
            "subject_id": sid,
            "face_images": face_imgs,
            "fingerprint_images": finger_imgs,
            "iris_images": iris_imgs
        })

    return pd.DataFrame(valid_data)

# ----------------------------
# 🔹 Collect subjects for each split
# ----------------------------
train_df = collect_subjects(train_ids)
val_df   = collect_subjects(val_ids)
test_df  = collect_subjects(test_ids)

# ----------------------------
# 🔹 Save splits as CSVs
# ----------------------------
save_path = "/kaggle/working/splits"
os.makedirs(save_path, exist_ok=True)

train_df.to_csv(os.path.join(save_path, "train_subjects.csv"), index=False)
val_df.to_csv(os.path.join(save_path, "val_subjects.csv"), index=False)
test_df.to_csv(os.path.join(save_path, "test_subjects.csv"), index=False)

print("\n✅ Splitting completed and saved!")
print(f"Train CSV: {os.path.join(save_path, 'train_subjects.csv')}")
print(f"Val CSV:   {os.path.join(save_path, 'val_subjects.csv')}")
print(f"Test CSV:  {os.path.join(save_path, 'test_subjects.csv')}")

# ================= NOTEBOOK CELL 2 =================
import pandas as pd
import random
import itertools
import os

random.seed(42)

SPLIT_DIR = "/kaggle/working/splits"
SAVE_DIR = "/kaggle/working/common_pairs"
os.makedirs(SAVE_DIR, exist_ok=True)

VAL_CSV = f"{SPLIT_DIR}/val_subjects.csv"
TEST_CSV = f"{SPLIT_DIR}/test_subjects.csv"

val_ids = pd.read_csv(VAL_CSV).iloc[:, 0].astype(str).tolist()
test_ids = pd.read_csv(TEST_CSV).iloc[:, 0].astype(str).tolist()

NUM_IMAGES = 10

def generate_pairs(subject_ids, num_genuine_per_subject=5, num_impostor=1500):
    pairs = []
    pair_id = 0

    for sid in subject_ids:
        all_combos = list(itertools.combinations(range(NUM_IMAGES), 2))  # 45 possible
        random.shuffle(all_combos)

        chosen = all_combos[:num_genuine_per_subject]
        for idx1, idx2 in chosen:
            pairs.append([pair_id, sid, sid, idx1, idx2, 1])
            pair_id += 1

    for _ in range(num_impostor):
        s1, s2 = random.sample(subject_ids, 2)
        idx1 = random.randint(0, NUM_IMAGES - 1)
        idx2 = random.randint(0, NUM_IMAGES - 1)
        pairs.append([pair_id, s1, s2, idx1, idx2, 0])
        pair_id += 1

    df = pd.DataFrame(
        pairs,
        columns=["pair_id", "subject1", "subject2", "idx1", "idx2", "label"]
    )
    return df

print("Generating validation pairs...")
val_pairs = generate_pairs(
    val_ids,
    num_genuine_per_subject=5,
    num_impostor=1500
)

print("Generating test pairs...")
test_pairs = generate_pairs(
    test_ids,
    num_genuine_per_subject=7,
    num_impostor=3000
)

val_save_path = f"{SAVE_DIR}/val_pairs_common.csv"
test_save_path = f"{SAVE_DIR}/test_pairs_common.csv"

val_pairs.to_csv(val_save_path, index=False)
test_pairs.to_csv(test_save_path, index=False)

print("\n✅ DONE")
print("Val pairs shape :", val_pairs.shape)
print("Test pairs shape:", test_pairs.shape)
print("\nSaved files:")
print(val_save_path)
print(test_save_path)

print("\nVal sample:")
print(val_pairs.head())

print("\nTest sample:")
print(test_pairs.head())
