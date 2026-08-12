"""
Dataset preparation utilities.

Expected dataset organization:

data/
├── face/
├── fingerprint/
└── iris/

The original public datasets are NOT included in this repository.
"""

from pathlib import Path
from typing import Dict, List


MODALITIES = ["face", "fingerprint", "iris"]

TRAIN_SUBJECTS = 480
VALIDATION_SUBJECTS = 60
TEST_SUBJECTS = 60


def list_files(folder: str,
               extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    """Return image files from a directory."""
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Dataset directory not found: {folder}")

    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def collect_modalities(data_root: str) -> Dict[str, List[str]]:
    """Collect image paths for the three biometric modalities."""
    root = Path(data_root)

    result = {}

    for modality in MODALITIES:
        result[modality] = [
            str(p) for p in list_files(root / modality)
        ]

    return result


def subject_disjoint_split(
    subjects,
    train_count=TRAIN_SUBJECTS,
    validation_count=VALIDATION_SUBJECTS,
    test_count=TEST_SUBJECTS
):
    """
    Create subject-disjoint train/validation/test lists.

    Subjects must be supplied as unique identifiers.
    """
    subjects = list(subjects)

    required = train_count + validation_count + test_count

    if len(subjects) < required:
        raise ValueError(
            f"At least {required} subjects are required; "
            f"received {len(subjects)}."
        )

    train = subjects[:train_count]

    validation = subjects[
        train_count:
        train_count + validation_count
    ]

    test = subjects[
        train_count + validation_count:
        required
    ]

    return {
        "train": train,
        "validation": validation,
        "test": test
    }


if __name__ == "__main__":
    print("Dataset preparation module loaded.")
    print("Modalities:", MODALITIES)
    print("Subject split:", {
        "train": TRAIN_SUBJECTS,
        "validation": VALIDATION_SUBJECTS,
        "test": TEST_SUBJECTS
    })
