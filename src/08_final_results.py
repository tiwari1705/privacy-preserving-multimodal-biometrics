"""Extracted from notebook3e7611700a(1).ipynb.
Cells preserved in original execution order.
Kaggle paths are intentionally preserved; replace them for local execution.
"""

# ================= NOTEBOOK CELL 89 =================
# ============================================================
# PAPER-READY TABLES + FIGURES
# CURRENT COMPLETED WORK ONLY
# - Unimodal plain cancellable
# - Unimodal true CKKS
# - Plain vs CKKS deviation
# - True CKKS runtime
# - True CKKS fusion
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# ============================================================
# 1. PATHS
# ============================================================

SAVE_ROOT = "/kaggle/working/final_paper_tables_current"
os.makedirs(SAVE_ROOT, exist_ok=True)

# True CKKS metric JSONs
FP_JSON   = "/kaggle/working/fingerprint_true_ckks_outputs/fingerprint_true_ckks_metrics.json"
FACE_JSON = "/kaggle/working/face_true_ckks_outputs/face_true_ckks_metrics.json"
IRIS_JSON = "/kaggle/working/iris_true_ckks_outputs/iris_true_ckks_metrics.json"

# Fusion summary
FUSION_SUMMARY_CSV = "/kaggle/working/fusion_true_ckks_outputs/fusion_true_ckks_summary.csv"

# ============================================================
# 2. LOAD
# ============================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

fp = load_json(FP_JSON)
face = load_json(FACE_JSON)
iris = load_json(IRIS_JSON)

fusion_df = pd.read_csv(FUSION_SUMMARY_CSV)

# ============================================================
# 3. BUILD PAPER TABLES
# ============================================================

# ---------- Table 1: Unimodal plain cancellable ----------
table1 = pd.DataFrame([
    {
        "Modality": "Fingerprint",
        "EER": fp["plain_eer"],
        "ROC-AUC": fp["plain_auc"],
        "Accuracy": fp["plain_accuracy"],
    },
    {
        "Modality": "Face",
        "EER": face["plain_eer"],
        "ROC-AUC": face["plain_auc"],
        "Accuracy": face["plain_accuracy"],
    },
    {
        "Modality": "Iris",
        "EER": iris["plain_eer"],
        "ROC-AUC": iris["plain_auc"],
        "Accuracy": iris["plain_accuracy"],
    },
])

# ---------- Table 2: Unimodal true CKKS ----------
table2 = pd.DataFrame([
    {
        "Modality": "Fingerprint",
        "EER": fp["ckks_eer"],
        "ROC-AUC": fp["ckks_auc"],
        "Accuracy": fp["ckks_accuracy"],
        "Precision": fp["precision"],
        "Recall": fp["recall"],
        "F1-score": fp["f1_score"],
    },
    {
        "Modality": "Face",
        "EER": face["ckks_eer"],
        "ROC-AUC": face["ckks_auc"],
        "Accuracy": face["ckks_accuracy"],
        "Precision": face["precision"],
        "Recall": face["recall"],
        "F1-score": face["f1_score"],
    },
    {
        "Modality": "Iris",
        "EER": iris["ckks_eer"],
        "ROC-AUC": iris["ckks_auc"],
        "Accuracy": iris["ckks_accuracy"],
        "Precision": iris["precision"],
        "Recall": iris["recall"],
        "F1-score": iris["f1_score"],
    },
])

# ---------- Table 3: Plain vs CKKS deviation ----------
table3 = pd.DataFrame([
    {
        "Modality": "Fingerprint",
        "Mean Abs. Diff.": fp["mean_abs_diff_vs_plain"],
        "Max Abs. Diff.": fp["max_abs_diff_vs_plain"],
    },
    {
        "Modality": "Face",
        "Mean Abs. Diff.": face["mean_abs_diff_vs_plain"],
        "Max Abs. Diff.": face["max_abs_diff_vs_plain"],
    },
    {
        "Modality": "Iris",
        "Mean Abs. Diff.": iris["mean_abs_diff_vs_plain"],
        "Max Abs. Diff.": iris["max_abs_diff_vs_plain"],
    },
])

# ---------- Table 4: Runtime ----------
table4 = pd.DataFrame([
    {
        "Modality": "Fingerprint",
        "Encryption Time (s)": fp["encryption_time_sec"],
        "Pair Scoring Time (s)": fp["pair_scoring_time_sec"],
        "Total Time (s)": fp["total_time_sec"],
    },
    {
        "Modality": "Face",
        "Encryption Time (s)": face["encryption_time_sec"],
        "Pair Scoring Time (s)": face["pair_scoring_time_sec"],
        "Total Time (s)": face["total_time_sec"],
    },
    {
        "Modality": "Iris",
        "Encryption Time (s)": iris["encryption_time_sec"],
        "Pair Scoring Time (s)": iris["pair_scoring_time_sec"],
        "Total Time (s)": iris["total_time_sec"],
    },
])

# ---------- Table 5: Fusion ----------
table5 = fusion_df.copy()
table5 = table5.rename(columns={
    "fusion_name": "Fusion",
    "weights": "Weights",
    "eer": "EER",
    "auc": "ROC-AUC",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score",
})
table5 = table5[["Fusion", "Weights", "EER", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1-score"]]

# ============================================================
# 4. ROUND FOR DISPLAY
# ============================================================

def rounded(df, sci=False):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if sci:
                df[col] = df[col].map(lambda x: f"{x:.4e}")
            else:
                df[col] = df[col].map(lambda x: f"{x:.4f}")
    return df

table1_disp = rounded(table1)
table2_disp = rounded(table2)
table3_disp = rounded(table3, sci=True)
table4_disp = rounded(table4)
table5_disp = rounded(table5)

# ============================================================
# 5. DISPLAY IN NOTEBOOK
# ============================================================

print("\n" + "="*80)
print("TABLE 1. UNIMODAL CANCELLABLE PERFORMANCE IN PLAIN DOMAIN")
print("="*80)
display(table1_disp)

print("\n" + "="*80)
print("TABLE 2. UNIMODAL CANCELLABLE PERFORMANCE UNDER TRUE CKKS")
print("="*80)
display(table2_disp)

print("\n" + "="*80)
print("TABLE 3. NUMERICAL DIFFERENCE BETWEEN PLAIN AND TRUE CKKS SCORES")
print("="*80)
display(table3_disp)

print("\n" + "="*80)
print("TABLE 4. TRUE CKKS RUNTIME OVERHEAD")
print("="*80)
display(table4_disp)

print("\n" + "="*80)
print("TABLE 5. TRUE CKKS FUSION PERFORMANCE")
print("="*80)
display(table5_disp)

# ============================================================
# 6. SAVE CSV TABLES
# ============================================================

table1.to_csv(os.path.join(SAVE_ROOT, "table1_unimodal_plain.csv"), index=False)
table2.to_csv(os.path.join(SAVE_ROOT, "table2_unimodal_true_ckks.csv"), index=False)
table3.to_csv(os.path.join(SAVE_ROOT, "table3_plain_vs_ckks_diff.csv"), index=False)
table4.to_csv(os.path.join(SAVE_ROOT, "table4_runtime_true_ckks.csv"), index=False)
table5.to_csv(os.path.join(SAVE_ROOT, "table5_fusion_true_ckks.csv"), index=False)

# also save formatted display tables
table1_disp.to_csv(os.path.join(SAVE_ROOT, "table1_unimodal_plain_display.csv"), index=False)
table2_disp.to_csv(os.path.join(SAVE_ROOT, "table2_unimodal_true_ckks_display.csv"), index=False)
table3_disp.to_csv(os.path.join(SAVE_ROOT, "table3_plain_vs_ckks_diff_display.csv"), index=False)
table4_disp.to_csv(os.path.join(SAVE_ROOT, "table4_runtime_true_ckks_display.csv"), index=False)
table5_disp.to_csv(os.path.join(SAVE_ROOT, "table5_fusion_true_ckks_display.csv"), index=False)

# ============================================================
# 7. PLOTS
# ============================================================

# ---------- Plot 1: Unimodal EER ----------
plt.figure(figsize=(7, 5))
plt.bar(table2["Modality"], table2["EER"])
plt.ylabel("EER")
plt.title("Unimodal Cancellable Performance Under True CKKS")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig1_unimodal_eer_true_ckks.png"), dpi=300, bbox_inches="tight")
plt.show()

# ---------- Plot 2: Unimodal ROC-AUC ----------
plt.figure(figsize=(7, 5))
plt.bar(table2["Modality"], table2["ROC-AUC"])
plt.ylabel("ROC-AUC")
plt.title("Unimodal ROC-AUC Under True CKKS")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig2_unimodal_auc_true_ckks.png"), dpi=300, bbox_inches="tight")
plt.show()

# ---------- Plot 3: Unimodal Accuracy ----------
plt.figure(figsize=(7, 5))
plt.bar(table2["Modality"], table2["Accuracy"])
plt.ylabel("Accuracy")
plt.title("Unimodal Accuracy Under True CKKS")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig3_unimodal_accuracy_true_ckks.png"), dpi=300, bbox_inches="tight")
plt.show()

# ---------- Plot 4: Runtime ----------
runtime_plot_df = table4.copy()
x = np.arange(len(runtime_plot_df))
w = 0.25

plt.figure(figsize=(8, 5))
plt.bar(x - w, runtime_plot_df["Encryption Time (s)"], width=w, label="Encryption")
plt.bar(x, runtime_plot_df["Pair Scoring Time (s)"], width=w, label="Pair Scoring")
plt.bar(x + w, runtime_plot_df["Total Time (s)"], width=w, label="Total")
plt.xticks(x, runtime_plot_df["Modality"])
plt.ylabel("Time (s)")
plt.title("Runtime Overhead of True CKKS Matching")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig4_runtime_true_ckks.png"), dpi=300, bbox_inches="tight")
plt.show()

# ---------- Plot 5: Fusion EER ----------
plt.figure(figsize=(8, 5))
plt.bar(table5["Fusion"], table5["EER"])
plt.ylabel("EER")
plt.title("Fusion Performance Under True CKKS")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig5_fusion_eer_true_ckks.png"), dpi=300, bbox_inches="tight")
plt.show()

# ---------- Plot 6: Fusion Accuracy ----------
plt.figure(figsize=(8, 5))
plt.bar(table5["Fusion"], table5["Accuracy"])
plt.ylabel("Accuracy")
plt.title("Fusion Accuracy Under True CKKS")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig6_fusion_accuracy_true_ckks.png"), dpi=300, bbox_inches="tight")
plt.show()

# ---------- Plot 7: Plain vs CKKS Score Difference ----------
plt.figure(figsize=(7, 5))
plt.bar(table3["Modality"], table3["Mean Abs. Diff."])
plt.ylabel("Mean Absolute Difference")
plt.title("Plain vs True CKKS Score Difference")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig7_plain_vs_ckks_diff.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 8. SAVE A SHORT RESULT SUMMARY TXT
# ============================================================

summary_text = """
TABLE 1: Unimodal cancellable performance in plain domain
TABLE 2: Unimodal cancellable performance under true CKKS
TABLE 3: Numerical difference between plain and true CKKS scores
TABLE 4: Runtime overhead of true CKKS matching
TABLE 5: True CKKS fusion performance

Figures saved:
fig1_unimodal_eer_true_ckks.png
fig2_unimodal_auc_true_ckks.png
fig3_unimodal_accuracy_true_ckks.png
fig4_runtime_true_ckks.png
fig5_fusion_eer_true_ckks.png
fig6_fusion_accuracy_true_ckks.png
fig7_plain_vs_ckks_diff.png
"""

with open(os.path.join(SAVE_ROOT, "paper_tables_summary.txt"), "w") as f:
    f.write(summary_text)

print("\n" + "="*80)
print("ALL PAPER-LEVEL TABLES AND FIGURES SAVED IN:")
print(SAVE_ROOT)
print("="*80)

# ================= NOTEBOOK CELL 96 =================
# ============================================================
# OPTION 1 -> PAPER-LEVEL TABLES + FIGURES
# Utility + Revocability + Cross-Key Unlinkability
# Uses the FIXED design-space results
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# ============================================================
# 1. PATHS
# ============================================================

ROOT = "/kaggle/working/design_space_revocability_unlinkability_fixed"
SAVE_ROOT = "/kaggle/working/option1_paper_tables"
os.makedirs(SAVE_ROOT, exist_ok=True)

FP_PATH   = os.path.join(ROOT, "fingerprint", "fingerprint_design_space_results.csv")
FACE_PATH = os.path.join(ROOT, "face", "face_design_space_results.csv")
IRIS_PATH = os.path.join(ROOT, "iris", "iris_design_space_results.csv")

# ============================================================
# 2. LOAD
# ============================================================

fp_df   = pd.read_csv(FP_PATH)
face_df = pd.read_csv(FACE_PATH)
iris_df = pd.read_csv(IRIS_PATH)

all_df = pd.concat([fp_df, face_df, iris_df], axis=0, ignore_index=True)

# ============================================================
# 3. HELPERS
# ============================================================

def pretty_modality(x):
    return str(x).capitalize()

def round_df(df, sci_cols=None):
    df = df.copy()
    sci_cols = sci_cols or []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if col in sci_cols:
                df[col] = df[col].map(lambda x: f"{x:.4e}")
            else:
                df[col] = df[col].map(lambda x: f"{x:.4f}")
    return df

def choose_best_utility(df_mod):
    same_df = df_mod[df_mod["comparison_type"] == "same_key_mean"].copy()
    return same_df.loc[same_df["eer"].idxmin()]

def choose_best_unlinkability(df_mod):
    cross_df = df_mod[df_mod["comparison_type"] == "cross_key_mean"].copy()
    cross_df["auc_dist_0_5"] = np.abs(cross_df["auc"] - 0.5)
    return cross_df.loc[cross_df["auc_dist_0_5"].idxmin()]

def choose_recommended(df_mod):
    same_df = df_mod[df_mod["comparison_type"] == "same_key_mean"].copy()
    cross_df = df_mod[df_mod["comparison_type"] == "cross_key_mean"].copy()

    merged = same_df.merge(
        cross_df,
        on=["modality", "dim", "binary"],
        suffixes=("_same", "_cross")
    )

    # composite score:
    # low same-key EER is good
    # cross-key AUC close to 0.5 is good
    # mean cross-key similarity close to 0 is good
    merged["same_eer_norm"] = merged["eer_same"] / merged["eer_same"].max()
    merged["auc_dist_norm"] = np.abs(merged["auc_cross"] - 0.5) / np.abs(merged["auc_cross"] - 0.5).max()
    merged["cross_sim_norm"] = np.abs(merged["mean_cross_key_similarity_cross"]) / np.abs(merged["mean_cross_key_similarity_cross"]).max()

    merged["composite_score"] = (
        0.50 * merged["same_eer_norm"] +
        0.30 * merged["auc_dist_norm"] +
        0.20 * merged["cross_sim_norm"]
    )

    return merged.loc[merged["composite_score"].idxmin()]

# ============================================================
# 4. TABLE 1 - SAME-KEY UTILITY TABLES
# ============================================================

same_key_table = all_df[all_df["comparison_type"] == "same_key_mean"].copy()
same_key_table["modality"] = same_key_table["modality"].map(pretty_modality)

table1 = same_key_table[[
    "modality", "dim", "binary", "eer", "auc", "accuracy", "f1"
]].copy()

table1 = table1.rename(columns={
    "modality": "Modality",
    "dim": "Dim",
    "binary": "Binary",
    "eer": "EER",
    "auc": "ROC-AUC",
    "accuracy": "Accuracy",
    "f1": "F1-score"
}).sort_values(["Modality", "Binary", "Dim"]).reset_index(drop=True)

table1_disp = round_df(table1)

print("\n" + "="*100)
print("TABLE 1. SAME-KEY UTILITY ACROSS DIMENSIONS AND TEMPLATE TYPES")
print("="*100)
display(table1_disp)

# ============================================================
# 5. TABLE 2 - CROSS-KEY UNLINKABILITY TABLES
# ============================================================

cross_key_table = all_df[all_df["comparison_type"] == "cross_key_mean"].copy()
cross_key_table["modality"] = cross_key_table["modality"].map(pretty_modality)

table2 = cross_key_table[[
    "modality", "dim", "binary", "eer", "auc", "accuracy", "f1",
    "mean_cross_key_similarity", "std_cross_key_similarity"
]].copy()

table2 = table2.rename(columns={
    "modality": "Modality",
    "dim": "Dim",
    "binary": "Binary",
    "eer": "Cross-key EER",
    "auc": "Cross-key ROC-AUC",
    "accuracy": "Cross-key Accuracy",
    "f1": "Cross-key F1-score",
    "mean_cross_key_similarity": "Mean Cross-key Similarity",
    "std_cross_key_similarity": "Std Cross-key Similarity"
}).sort_values(["Modality", "Binary", "Dim"]).reset_index(drop=True)

table2_disp = round_df(table2)

print("\n" + "="*100)
print("TABLE 2. CROSS-KEY UNLINKABILITY ACROSS DIMENSIONS AND TEMPLATE TYPES")
print("="*100)
display(table2_disp)

# ============================================================
# 6. TABLE 3 - BEST UTILITY / BEST UNLINKABILITY / RECOMMENDED
# ============================================================

summary_rows = []

for mod in ["fingerprint", "face", "iris"]:
    df_mod = all_df[all_df["modality"] == mod].copy()

    best_u = choose_best_utility(df_mod)
    best_x = choose_best_unlinkability(df_mod)
    rec    = choose_recommended(df_mod)

    summary_rows.append({
        "Modality": pretty_modality(mod),

        "Best Utility Dim": int(best_u["dim"]),
        "Best Utility Binary": bool(best_u["binary"]),
        "Best Utility EER": float(best_u["eer"]),
        "Best Utility ROC-AUC": float(best_u["auc"]),
        "Best Utility Accuracy": float(best_u["accuracy"]),

        "Best Unlinkability Dim": int(best_x["dim"]),
        "Best Unlinkability Binary": bool(best_x["binary"]),
        "Best Unlinkability Cross-key EER": float(best_x["eer"]),
        "Best Unlinkability Cross-key ROC-AUC": float(best_x["auc"]),
        "Best Unlinkability Mean Similarity": float(best_x["mean_cross_key_similarity"]),

        "Recommended Dim": int(rec["dim"]),
        "Recommended Binary": bool(rec["binary"]),
        "Recommended Same-key EER": float(rec["eer_same"]),
        "Recommended Cross-key ROC-AUC": float(rec["auc_cross"]),
        "Recommended Mean Cross-key Similarity": float(rec["mean_cross_key_similarity_cross"]),
    })

table3 = pd.DataFrame(summary_rows)
table3_disp = round_df(table3)

print("\n" + "="*100)
print("TABLE 3. BEST UTILITY, BEST UNLINKABILITY, AND RECOMMENDED CONFIGURATION")
print("="*100)
display(table3_disp)

# ============================================================
# 7. SAVE TABLES
# ============================================================

table1.to_csv(os.path.join(SAVE_ROOT, "table1_samekey_utility.csv"), index=False)
table2.to_csv(os.path.join(SAVE_ROOT, "table2_crosskey_unlinkability.csv"), index=False)
table3.to_csv(os.path.join(SAVE_ROOT, "table3_best_recommended_configs.csv"), index=False)

table1_disp.to_csv(os.path.join(SAVE_ROOT, "table1_samekey_utility_display.csv"), index=False)
table2_disp.to_csv(os.path.join(SAVE_ROOT, "table2_crosskey_unlinkability_display.csv"), index=False)
table3_disp.to_csv(os.path.join(SAVE_ROOT, "table3_best_recommended_configs_display.csv"), index=False)

# ============================================================
# 8. FIGURES - SAME-KEY EER VS DIM
# ============================================================

for mod in ["fingerprint", "face", "iris"]:
    df_mod = all_df[(all_df["modality"] == mod) & (all_df["comparison_type"] == "same_key_mean")].copy()

    plt.figure(figsize=(8, 5))
    for binary in sorted(df_mod["binary"].unique()):
        sub = df_mod[df_mod["binary"] == binary]
        plt.plot(sub["dim"], sub["eer"], marker="o", label=f"binary={binary}")
    plt.xlabel("Projection Dimension")
    plt.ylabel("Same-key EER")
    plt.title(f"{pretty_modality(mod)}: Same-key Utility vs Dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, f"fig_{mod}_samekey_eer_vs_dim.png"), dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================
# 9. FIGURES - CROSS-KEY EER VS DIM
# ============================================================

for mod in ["fingerprint", "face", "iris"]:
    df_mod = all_df[(all_df["modality"] == mod) & (all_df["comparison_type"] == "cross_key_mean")].copy()

    plt.figure(figsize=(8, 5))
    for binary in sorted(df_mod["binary"].unique()):
        sub = df_mod[df_mod["binary"] == binary]
        plt.plot(sub["dim"], sub["eer"], marker="o", label=f"binary={binary}")
    plt.xlabel("Projection Dimension")
    plt.ylabel("Cross-key EER")
    plt.title(f"{pretty_modality(mod)}: Cross-key Unlinkability vs Dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, f"fig_{mod}_crosskey_eer_vs_dim.png"), dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================
# 10. FIGURES - CROSS-KEY MEAN SIMILARITY VS DIM
# ============================================================

for mod in ["fingerprint", "face", "iris"]:
    df_mod = all_df[(all_df["modality"] == mod) & (all_df["comparison_type"] == "cross_key_mean")].copy()

    plt.figure(figsize=(8, 5))
    for binary in sorted(df_mod["binary"].unique()):
        sub = df_mod[df_mod["binary"] == binary]
        plt.plot(sub["dim"], sub["mean_cross_key_similarity"], marker="o", label=f"binary={binary}")
    plt.xlabel("Projection Dimension")
    plt.ylabel("Mean Cross-key Cosine Similarity")
    plt.title(f"{pretty_modality(mod)}: Cross-key Similarity vs Dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, f"fig_{mod}_crosskey_similarity_vs_dim.png"), dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================
# 11. COMBINED FIGURE - BEST SAME-KEY EER
# ============================================================

best_same_rows = []
for mod in ["fingerprint", "face", "iris"]:
    df_mod = all_df[all_df["modality"] == mod].copy()
    best_u = choose_best_utility(df_mod)
    best_same_rows.append({
        "Modality": pretty_modality(mod),
        "EER": float(best_u["eer"])
    })

best_same_df = pd.DataFrame(best_same_rows)

plt.figure(figsize=(7, 5))
plt.bar(best_same_df["Modality"], best_same_df["EER"])
plt.ylabel("Best Same-key EER")
plt.title("Best Utility Configuration per Modality")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig_best_samekey_eer_per_modality.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 12. SHORT SUMMARY TXT
# ============================================================

summary_text = """
OPTION 1 PAPER TABLES GENERATED

Table 1: Same-key utility across dimensions and template types
Table 2: Cross-key unlinkability across dimensions and template types
Table 3: Best utility, best unlinkability, and recommended configuration

Figures saved:
- same-key EER vs dimension (per modality)
- cross-key EER vs dimension (per modality)
- cross-key similarity vs dimension (per modality)
- best same-key EER per modality
"""

with open(os.path.join(SAVE_ROOT, "option1_tables_summary.txt"), "w") as f:
    f.write(summary_text)

print("\n" + "="*100)
print("ALL OPTION 1 PAPER TABLES AND FIGURES SAVED IN:")
print(SAVE_ROOT)
print("="*100)

# ================= NOTEBOOK CELL 97 =================
import os
import pandas as pd
import matplotlib.pyplot as plt

out_dir = "/mnt/data/paper_charts_final"
os.makedirs(out_dir, exist_ok=True)

# Final results from the user's verified runs
unimodal = pd.DataFrame([
    {"Modality": "Fingerprint", "Domain": "Plain Cancellable", "EER": 0.1685185185, "AUC": 0.9078251029, "Accuracy": 0.8312962963},
    {"Modality": "Fingerprint", "Domain": "True CKKS",         "EER": 0.1685185185, "AUC": 0.9078251029, "Accuracy": 0.8312962963},
    {"Modality": "Face",        "Domain": "Plain Cancellable", "EER": 0.0566666667, "AUC": 0.9852945130, "Accuracy": 0.9437037037},
    {"Modality": "Face",        "Domain": "True CKKS",         "EER": 0.0566666667, "AUC": 0.9852945130, "Accuracy": 0.9437037037},
    {"Modality": "Iris",        "Domain": "Plain Cancellable", "EER": 0.0177777778, "AUC": 0.9979068587, "Accuracy": 0.9818518519},
    {"Modality": "Iris",        "Domain": "True CKKS",         "EER": 0.0177777778, "AUC": 0.9979068587, "Accuracy": 0.9818518519},
])

fusion = pd.DataFrame([
    {"Fusion": "Face + Fingerprint", "EER": 0.0381481481, "AUC": 0.9932150892, "Accuracy": 0.9616666667},
    {"Fusion": "Face + Iris",        "EER": 0.0037037037, "AUC": 0.9999008230, "Accuracy": 0.9962962963},
    {"Fusion": "Fingerprint + Iris", "EER": 0.0122222222, "AUC": 0.9992325103, "Accuracy": 0.9877777778},
    {"Fusion": "All Three",          "EER": 0.0029629630, "AUC": 0.9999507545, "Accuracy": 0.9972222222},
])

# Save CSVs
unimodal.to_csv(os.path.join(out_dir, "unimodal_results.csv"), index=False)
fusion.to_csv(os.path.join(out_dir, "fusion_results.csv"), index=False)

# 1) Unimodal EER paired bar chart
mods = ["Fingerprint", "Face", "Iris"]
plain_eer = [unimodal[(unimodal["Modality"] == m) & (unimodal["Domain"] == "Plain Cancellable")]["EER"].iloc[0] for m in mods]
ckks_eer = [unimodal[(unimodal["Modality"] == m) & (unimodal["Domain"] == "True CKKS")]["EER"].iloc[0] for m in mods]

x = range(len(mods))
w = 0.35
plt.figure(figsize=(8, 5))
plt.bar([i - w/2 for i in x], plain_eer, width=w, label="Plain Cancellable")
plt.bar([i + w/2 for i in x], ckks_eer, width=w, label="True CKKS")
plt.xticks(list(x), mods)
plt.ylabel("EER")
plt.title("Unimodal Verification: Plain vs True CKKS")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig1_unimodal_eer_plain_vs_ckks.png"), dpi=300, bbox_inches="tight")
plt.close()

# 2) Unimodal Accuracy paired bar chart
plain_acc = [unimodal[(unimodal["Modality"] == m) & (unimodal["Domain"] == "Plain Cancellable")]["Accuracy"].iloc[0] for m in mods]
ckks_acc = [unimodal[(unimodal["Modality"] == m) & (unimodal["Domain"] == "True CKKS")]["Accuracy"].iloc[0] for m in mods]

plt.figure(figsize=(8, 5))
plt.bar([i - w/2 for i in x], plain_acc, width=w, label="Plain Cancellable")
plt.bar([i + w/2 for i in x], ckks_acc, width=w, label="True CKKS")
plt.xticks(list(x), mods)
plt.ylabel("Accuracy")
plt.title("Unimodal Accuracy: Plain vs True CKKS")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig2_unimodal_accuracy_plain_vs_ckks.png"), dpi=300, bbox_inches="tight")
plt.close()

# 3) Fusion EER bar chart
plt.figure(figsize=(8.5, 5))
plt.bar(fusion["Fusion"], fusion["EER"])
plt.xticks(rotation=20, ha="right")
plt.ylabel("EER")
plt.title("Multimodal Fusion Performance (True CKKS)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig3_fusion_eer_true_ckks.png"), dpi=300, bbox_inches="tight")
plt.close()

# 4) Fusion Accuracy bar chart
plt.figure(figsize=(8.5, 5))
plt.bar(fusion["Fusion"], fusion["Accuracy"])
plt.xticks(rotation=20, ha="right")
plt.ylabel("Accuracy")
plt.title("Multimodal Fusion Accuracy (True CKKS)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig4_fusion_accuracy_true_ckks.png"), dpi=300, bbox_inches="tight")
plt.close()

# 5) Best combined summary chart: EER vs Accuracy scatter
summary_points = pd.DataFrame([
    {"Label": "FP Plain", "Category": "Unimodal", "EER": 0.1685185185, "Accuracy": 0.8312962963},
    {"Label": "Face Plain", "Category": "Unimodal", "EER": 0.0566666667, "Accuracy": 0.9437037037},
    {"Label": "Iris Plain", "Category": "Unimodal", "EER": 0.0177777778, "Accuracy": 0.9818518519},
    {"Label": "Face+FP", "Category": "Fusion", "EER": 0.0381481481, "Accuracy": 0.9616666667},
    {"Label": "Face+Iris", "Category": "Fusion", "EER": 0.0037037037, "Accuracy": 0.9962962963},
    {"Label": "FP+Iris", "Category": "Fusion", "EER": 0.0122222222, "Accuracy": 0.9877777778},
    {"Label": "All Three", "Category": "Fusion", "EER": 0.0029629630, "Accuracy": 0.9972222222},
])
summary_points.to_csv(os.path.join(out_dir, "summary_points.csv"), index=False)

plt.figure(figsize=(8, 5.5))
for _, row in summary_points.iterrows():
    plt.scatter(row["EER"], row["Accuracy"], s=80)
    plt.annotate(row["Label"], (row["EER"], row["Accuracy"]), textcoords="offset points", xytext=(5,5))
plt.xlabel("EER")
plt.ylabel("Accuracy")
plt.title("Overall Summary: Utility Gain from Multimodal Fusion")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig5_summary_scatter_eer_vs_accuracy.png"), dpi=300, bbox_inches="tight")
plt.close()

# 6) Paper-friendly ranking chart (lower is better) for easiest interpretation
rank_df = pd.DataFrame([
    {"System": "Fingerprint", "Type": "Unimodal", "EER": 0.1685185185},
    {"System": "Face", "Type": "Unimodal", "EER": 0.0566666667},
    {"System": "Iris", "Type": "Unimodal", "EER": 0.0177777778},
    {"System": "Face + Fingerprint", "Type": "Fusion", "EER": 0.0381481481},
    {"System": "Face + Iris", "Type": "Fusion", "EER": 0.0037037037},
    {"System": "Fingerprint + Iris", "Type": "Fusion", "EER": 0.0122222222},
    {"System": "Face + Fingerprint + Iris", "Type": "Fusion", "EER": 0.0029629630},
]).sort_values("EER", ascending=True)

plt.figure(figsize=(9, 5.5))
plt.barh(rank_df["System"], rank_df["EER"])
plt.xlabel("EER")
plt.ylabel("System")
plt.title("System Ranking by EER (Lower is Better)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig6_system_ranking_by_eer.png"), dpi=300, bbox_inches="tight")
plt.close()

# Create a captions text file
captions = """Figure 1. Unimodal verification performance in terms of EER for plain cancellable and true CKKS matching.
Figure 2. Unimodal verification accuracy for plain cancellable and true CKKS matching.
Figure 3. EER of bimodal and trimodal fusion under true CKKS matching.
Figure 4. Accuracy of bimodal and trimodal fusion under true CKKS matching.
Figure 5. Overall summary scatter plot showing the utility gain from multimodal fusion in terms of EER and accuracy.
Figure 6. System ranking based on EER, illustrating the relative gains from multimodal fusion over unimodal verification.
"""
with open(os.path.join(out_dir, "figure_captions.txt"), "w") as f:
    f.write(captions)

print("Saved files in:", out_dir)
print("\nCreated:")
for name in sorted(os.listdir(out_dir)):
    print("-", name)


# ================= NOTEBOOK CELL 98 =================
# ============================================================
# PAPER-FRIENDLY CHARTS: SHOW ON KAGGLE SCREEN + SAVE PNG
# Current final verified results
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

SAVE_ROOT = "/kaggle/working/paper_charts_screen"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ============================================================
# 1. FINAL VERIFIED RESULTS
# ============================================================

unimodal = pd.DataFrame([
    {"Modality": "Fingerprint", "Domain": "Plain Cancellable", "EER": 0.1685185185, "AUC": 0.9078251029, "Accuracy": 0.8312962963},
    {"Modality": "Fingerprint", "Domain": "True CKKS",         "EER": 0.1685185185, "AUC": 0.9078251029, "Accuracy": 0.8312962963},

    {"Modality": "Face",        "Domain": "Plain Cancellable", "EER": 0.0566666667, "AUC": 0.9852945130, "Accuracy": 0.9437037037},
    {"Modality": "Face",        "Domain": "True CKKS",         "EER": 0.0566666667, "AUC": 0.9852945130, "Accuracy": 0.9437037037},

    {"Modality": "Iris",        "Domain": "Plain Cancellable", "EER": 0.0177777778, "AUC": 0.9979068587, "Accuracy": 0.9818518519},
    {"Modality": "Iris",        "Domain": "True CKKS",         "EER": 0.0177777778, "AUC": 0.9979068587, "Accuracy": 0.9818518519},
])

fusion = pd.DataFrame([
    {"Fusion": "Face + Fingerprint",          "EER": 0.0381481481, "AUC": 0.9932150892, "Accuracy": 0.9616666667},
    {"Fusion": "Face + Iris",                 "EER": 0.0037037037, "AUC": 0.9999008230, "Accuracy": 0.9962962963},
    {"Fusion": "Fingerprint + Iris",          "EER": 0.0122222222, "AUC": 0.9992325103, "Accuracy": 0.9877777778},
    {"Fusion": "Face + Fingerprint + Iris",   "EER": 0.0029629630, "AUC": 0.9999507545, "Accuracy": 0.9972222222},
])

# save CSV too
unimodal.to_csv(os.path.join(SAVE_ROOT, "unimodal_results.csv"), index=False)
fusion.to_csv(os.path.join(SAVE_ROOT, "fusion_results.csv"), index=False)

# ============================================================
# 2. FIGURE 1 - UNIMODAL EER (PLAIN VS TRUE CKKS)
# ============================================================

mods = ["Fingerprint", "Face", "Iris"]
plain_eer = [unimodal[(unimodal["Modality"] == m) & (unimodal["Domain"] == "Plain Cancellable")]["EER"].iloc[0] for m in mods]
ckks_eer  = [unimodal[(unimodal["Modality"] == m) & (unimodal["Domain"] == "True CKKS")]["EER"].iloc[0] for m in mods]

x = range(len(mods))
w = 0.35

plt.figure(figsize=(8, 5))
plt.bar([i - w/2 for i in x], plain_eer, width=w, label="Plain Cancellable")
plt.bar([i + w/2 for i in x], ckks_eer, width=w, label="True CKKS")
plt.xticks(list(x), mods)
plt.ylabel("EER")
plt.title("Unimodal Verification: Plain vs True CKKS")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig1_unimodal_eer_plain_vs_ckks.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 3. FIGURE 2 - UNIMODAL ACCURACY (PLAIN VS TRUE CKKS)
# ============================================================

plain_acc = [unimodal[(unimodal["Modality"] == m) & (unimodal["Domain"] == "Plain Cancellable")]["Accuracy"].iloc[0] for m in mods]
ckks_acc  = [unimodal[(unimodal["Modality"] == m) & (unimodal["Domain"] == "True CKKS")]["Accuracy"].iloc[0] for m in mods]

plt.figure(figsize=(8, 5))
plt.bar([i - w/2 for i in x], plain_acc, width=w, label="Plain Cancellable")
plt.bar([i + w/2 for i in x], ckks_acc, width=w, label="True CKKS")
plt.xticks(list(x), mods)
plt.ylabel("Accuracy")
plt.title("Unimodal Accuracy: Plain vs True CKKS")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig2_unimodal_accuracy_plain_vs_ckks.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 4. FIGURE 3 - FUSION EER
# ============================================================

plt.figure(figsize=(9, 5))
plt.bar(fusion["Fusion"], fusion["EER"])
plt.xticks(rotation=20, ha="right")
plt.ylabel("EER")
plt.title("Multimodal Fusion Performance (True CKKS)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig3_fusion_eer_true_ckks.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 5. FIGURE 4 - FUSION ACCURACY
# ============================================================

plt.figure(figsize=(9, 5))
plt.bar(fusion["Fusion"], fusion["Accuracy"])
plt.xticks(rotation=20, ha="right")
plt.ylabel("Accuracy")
plt.title("Multimodal Fusion Accuracy (True CKKS)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig4_fusion_accuracy_true_ckks.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 6. FIGURE 5 - SUMMARY SCATTER (BEST OVERALL CHART)
# ============================================================

summary_points = pd.DataFrame([
    {"Label": "Fingerprint", "Category": "Unimodal", "EER": 0.1685185185, "Accuracy": 0.8312962963},
    {"Label": "Face", "Category": "Unimodal", "EER": 0.0566666667, "Accuracy": 0.9437037037},
    {"Label": "Iris", "Category": "Unimodal", "EER": 0.0177777778, "Accuracy": 0.9818518519},
    {"Label": "Face + Fingerprint", "Category": "Fusion", "EER": 0.0381481481, "Accuracy": 0.9616666667},
    {"Label": "Face + Iris", "Category": "Fusion", "EER": 0.0037037037, "Accuracy": 0.9962962963},
    {"Label": "Fingerprint + Iris", "Category": "Fusion", "EER": 0.0122222222, "Accuracy": 0.9877777778},
    {"Label": "All Three", "Category": "Fusion", "EER": 0.0029629630, "Accuracy": 0.9972222222},
])

summary_points.to_csv(os.path.join(SAVE_ROOT, "summary_points.csv"), index=False)

plt.figure(figsize=(8.5, 5.5))
for _, row in summary_points.iterrows():
    plt.scatter(row["EER"], row["Accuracy"], s=90)
    plt.annotate(row["Label"], (row["EER"], row["Accuracy"]), textcoords="offset points", xytext=(5,5))
plt.xlabel("EER")
plt.ylabel("Accuracy")
plt.title("Overall Summary: Utility Gain from Multimodal Fusion")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig5_summary_scatter_eer_vs_accuracy.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 7. FIGURE 6 - SYSTEM RANKING BY EER
# ============================================================

rank_df = pd.DataFrame([
    {"System": "Fingerprint", "Type": "Unimodal", "EER": 0.1685185185},
    {"System": "Face", "Type": "Unimodal", "EER": 0.0566666667},
    {"System": "Iris", "Type": "Unimodal", "EER": 0.0177777778},
    {"System": "Face + Fingerprint", "Type": "Fusion", "EER": 0.0381481481},
    {"System": "Face + Iris", "Type": "Fusion", "EER": 0.0037037037},
    {"System": "Fingerprint + Iris", "Type": "Fusion", "EER": 0.0122222222},
    {"System": "Face + Fingerprint + Iris", "Type": "Fusion", "EER": 0.0029629630},
]).sort_values("EER", ascending=True)

plt.figure(figsize=(9, 5.5))
plt.barh(rank_df["System"], rank_df["EER"])
plt.xlabel("EER")
plt.ylabel("System")
plt.title("System Ranking by EER (Lower is Better)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig6_system_ranking_by_eer.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 8. OPTIONAL - PRINT SIMPLE CAPTIONS
# ============================================================

captions = {
    "Fig 1": "Unimodal verification performance in terms of EER for plain cancellable and true CKKS matching.",
    "Fig 2": "Unimodal verification accuracy for plain cancellable and true CKKS matching.",
    "Fig 3": "EER of bimodal and trimodal fusion under true CKKS matching.",
    "Fig 4": "Accuracy of bimodal and trimodal fusion under true CKKS matching.",
    "Fig 5": "Overall summary showing the utility gain from multimodal fusion in terms of EER and accuracy.",
    "Fig 6": "System ranking based on EER, illustrating the gains achieved by multimodal fusion over unimodal verification."
}

print("\nFigure captions:")
for k, v in captions.items():
    print(f"{k}: {v}")

print("\nAll charts saved in:", SAVE_ROOT)

# ================= NOTEBOOK CELL 99 =================
# ============================================================
# COMBINED ROC CURVE
# Best Unimodal Plain vs Encrypted
# Best Fusion Plain vs Encrypted
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

SAVE_ROOT = "/kaggle/working/paper_best_roc"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ------------------------------------------------------------
# 1. INPUT PATHS
# ------------------------------------------------------------

# Best unimodal = Iris
IRIS_PLAIN_PATH = "/kaggle/working/iris_cancellable_outputs_trueckks/iris_test_scores_cancellable_plain.csv"
IRIS_CKKS_PATH  = "/kaggle/working/iris_true_ckks_outputs/iris_test_scores_true_ckks.csv"

# Plain score files for fusion rebuild
FP_PLAIN_PATH   = "/kaggle/working/fingerprint_cancellable_outputs_trueckks/fingerprint_test_scores_cancellable_plain.csv"
FACE_PLAIN_PATH = "/kaggle/working/face_cancellable_outputs_trueckks/face_test_scores_cancellable_plain.csv"
IRIS_PLAIN_PATH2= "/kaggle/working/iris_cancellable_outputs_trueckks/iris_test_scores_cancellable_plain.csv"

# Encrypted score files for fusion rebuild
FP_CKKS_PATH    = "/kaggle/working/fingerprint_true_ckks_outputs/fingerprint_test_scores_true_ckks.csv"
FACE_CKKS_PATH  = "/kaggle/working/face_true_ckks_outputs/face_test_scores_true_ckks.csv"
IRIS_CKKS_PATH2 = "/kaggle/working/iris_true_ckks_outputs/iris_test_scores_true_ckks.csv"

# ------------------------------------------------------------
# 2. HELPERS
# ------------------------------------------------------------

def load_scores(path, score_name):
    df = pd.read_csv(path)
    keep_cols = ["pair_id", "subject1", "subject2", "idx1", "idx2", "label", "score"]
    df = df[keep_cols].copy()
    return df.rename(columns={"score": score_name})

def normalize_scores(x):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def fuse_all_three(face_path, fp_path, iris_path, mode_name):
    face_df = load_scores(face_path, "score_face")
    fp_df   = load_scores(fp_path, "score_fingerprint")
    iris_df = load_scores(iris_path, "score_iris")

    merge_keys = ["pair_id", "subject1", "subject2", "idx1", "idx2", "label"]
    df = face_df.merge(fp_df, on=merge_keys).merge(iris_df, on=merge_keys)

    df["score_face_norm"] = normalize_scores(df["score_face"])
    df["score_fingerprint_norm"] = normalize_scores(df["score_fingerprint"])
    df["score_iris_norm"] = normalize_scores(df["score_iris"])

    # validated weights
    w_face, w_fp, w_iris = 0.30, 0.20, 0.50

    df["fused_score"] = (
        w_face * df["score_face_norm"] +
        w_fp   * df["score_fingerprint_norm"] +
        w_iris * df["score_iris_norm"]
    )

    return df

def get_roc(df, score_col="score"):
    y_true = df["label"].astype(int).values
    y_score = df[score_col].astype(float).values
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return fpr, tpr, auc

# ------------------------------------------------------------
# 3. LOAD BEST UNIMODAL
# ------------------------------------------------------------

iris_plain_df = pd.read_csv(IRIS_PLAIN_PATH)
iris_ckks_df  = pd.read_csv(IRIS_CKKS_PATH)

fpr_iris_plain, tpr_iris_plain, auc_iris_plain = get_roc(iris_plain_df, "score")
fpr_iris_ckks,  tpr_iris_ckks,  auc_iris_ckks  = get_roc(iris_ckks_df, "score")

# ------------------------------------------------------------
# 4. BUILD BEST FUSION (ALL THREE)
# ------------------------------------------------------------

fusion_plain_df = fuse_all_three(FACE_PLAIN_PATH, FP_PLAIN_PATH, IRIS_PLAIN_PATH2, "plain")
fusion_ckks_df  = fuse_all_three(FACE_CKKS_PATH,  FP_CKKS_PATH,  IRIS_CKKS_PATH2,  "ckks")

fpr_fusion_plain, tpr_fusion_plain, auc_fusion_plain = get_roc(fusion_plain_df, "fused_score")
fpr_fusion_ckks,  tpr_fusion_ckks,  auc_fusion_ckks  = get_roc(fusion_ckks_df, "fused_score")

# save fusion score files too
fusion_plain_df.to_csv(os.path.join(SAVE_ROOT, "all_three_plain_fusion_scores.csv"), index=False)
fusion_ckks_df.to_csv(os.path.join(SAVE_ROOT, "all_three_trueckks_fusion_scores.csv"), index=False)

# ------------------------------------------------------------
# 5. COMBINED ROC FIGURE
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.plot(fpr_iris_plain,  tpr_iris_plain,  label=f"Iris Plain (AUC={auc_iris_plain:.4f})")
plt.plot(fpr_iris_ckks,   tpr_iris_ckks,   label=f"Iris True CKKS (AUC={auc_iris_ckks:.4f})")
plt.plot(fpr_fusion_plain,tpr_fusion_plain,label=f"All-Three Plain Fusion (AUC={auc_fusion_plain:.4f})")
plt.plot(fpr_fusion_ckks, tpr_fusion_ckks, label=f"All-Three True CKKS Fusion (AUC={auc_fusion_ckks:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Combined ROC: Plain vs Encrypted, Unimodal vs Fusion")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "combined_roc_best_systems.png"), dpi=300, bbox_inches="tight")
plt.show()

print("Saved in:", SAVE_ROOT)

# ================= NOTEBOOK CELL 100 =================
# ============================================================
# OPTION 1 BEST PAPER CHARTS
# Heatmaps + Tradeoff Scatter + Summary Bar
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SAVE_ROOT = "/kaggle/working/option1_best_graphs"
os.makedirs(SAVE_ROOT, exist_ok=True)

ROOT = "/kaggle/working/design_space_revocability_unlinkability_fixed"

FP_PATH   = os.path.join(ROOT, "fingerprint", "fingerprint_design_space_results.csv")
FACE_PATH = os.path.join(ROOT, "face", "face_design_space_results.csv")
IRIS_PATH = os.path.join(ROOT, "iris", "iris_design_space_results.csv")

fp_df   = pd.read_csv(FP_PATH)
face_df = pd.read_csv(FACE_PATH)
iris_df = pd.read_csv(IRIS_PATH)

all_df = pd.concat([fp_df, face_df, iris_df], axis=0, ignore_index=True)

def pretty_modality(x):
    return str(x).capitalize()

def draw_heatmap(ax, matrix, row_labels, col_labels, title):
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def modality_heatmaps(modality):
    df_mod = all_df[all_df["modality"] == modality].copy()

    same_df = df_mod[df_mod["comparison_type"] == "same_key_mean"].copy()
    cross_df = df_mod[df_mod["comparison_type"] == "cross_key_mean"].copy()

    dims = [64, 128, 256, 512, 1024]
    binaries = [False, True]
    col_labels = ["Non-binary", "Binary"]

    # Same-key EER matrix
    same_eer_mat = []
    cross_eer_mat = []
    cross_auc_mat = []

    for d in dims:
        same_row = []
        cross_row_eer = []
        cross_row_auc = []
        for b in binaries:
            same_val = same_df[(same_df["dim"] == d) & (same_df["binary"] == b)]["eer"].iloc[0]
            cross_val_eer = cross_df[(cross_df["dim"] == d) & (cross_df["binary"] == b)]["eer"].iloc[0]
            cross_val_auc = cross_df[(cross_df["dim"] == d) & (cross_df["binary"] == b)]["auc"].iloc[0]

            same_row.append(same_val)
            cross_row_eer.append(cross_val_eer)
            cross_row_auc.append(cross_val_auc)

        same_eer_mat.append(same_row)
        cross_eer_mat.append(cross_row_eer)
        cross_auc_mat.append(cross_row_auc)

    same_eer_mat = np.array(same_eer_mat)
    cross_eer_mat = np.array(cross_eer_mat)
    cross_auc_mat = np.array(cross_auc_mat)

    # ---- Figure A: same-key EER heatmap ----
    fig, ax = plt.subplots(figsize=(6, 5))
    draw_heatmap(ax, same_eer_mat, dims, col_labels, f"{pretty_modality(modality)}: Same-key EER")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, f"{modality}_samekey_eer_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # ---- Figure B: cross-key EER heatmap ----
    fig, ax = plt.subplots(figsize=(6, 5))
    draw_heatmap(ax, cross_eer_mat, dims, col_labels, f"{pretty_modality(modality)}: Cross-key EER")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, f"{modality}_crosskey_eer_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # ---- Figure C: cross-key AUC heatmap ----
    fig, ax = plt.subplots(figsize=(6, 5))
    draw_heatmap(ax, cross_auc_mat, dims, col_labels, f"{pretty_modality(modality)}: Cross-key ROC-AUC")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, f"{modality}_crosskey_auc_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.show()

for mod in ["fingerprint", "face", "iris"]:
    modality_heatmaps(mod)

# ------------------------------------------------------------
# Utility-Unlinkability tradeoff scatter
# x = same-key EER
# y = |cross-key AUC - 0.5|
# smaller is better in both
# ------------------------------------------------------------

same_df = all_df[all_df["comparison_type"] == "same_key_mean"].copy()
cross_df = all_df[all_df["comparison_type"] == "cross_key_mean"].copy()

merged = same_df.merge(
    cross_df,
    on=["modality", "dim", "binary"],
    suffixes=("_same", "_cross")
)

merged["unlinkability_distance"] = np.abs(merged["auc_cross"] - 0.5)

plt.figure(figsize=(8, 6))
for _, row in merged.iterrows():
    label = f"{pretty_modality(row['modality'])}-{int(row['dim'])}-{'B' if row['binary'] else 'NB'}"
    plt.scatter(row["eer_same"], row["unlinkability_distance"], s=80)
    plt.annotate(label, (row["eer_same"], row["unlinkability_distance"]), textcoords="offset points", xytext=(4,4), fontsize=8)

plt.xlabel("Same-key EER (lower is better)")
plt.ylabel("|Cross-key AUC - 0.5| (lower is better)")
plt.title("Utility–Unlinkability Tradeoff Across Protected Configurations")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "utility_unlinkability_tradeoff_scatter.png"), dpi=300, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# Summary bar chart: best utility per modality
# ------------------------------------------------------------

best_rows = []
for mod in ["fingerprint", "face", "iris"]:
    sub = same_df[same_df["modality"] == mod].copy()
    best = sub.loc[sub["eer"].idxmin()]
    best_rows.append({
        "Modality": pretty_modality(mod),
        "Best Same-key EER": best["eer"]
    })

best_df = pd.DataFrame(best_rows)

plt.figure(figsize=(7, 5))
plt.bar(best_df["Modality"], best_df["Best Same-key EER"])
plt.ylabel("Best Same-key EER")
plt.title("Best Utility Configuration per Modality")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "best_samekey_eer_per_modality.png"), dpi=300, bbox_inches="tight")
plt.show()

print("Saved in:", SAVE_ROOT)

# ================= NOTEBOOK CELL 101 =================
# ============================================================
# CLEAN IEEE-STYLE FIGURES FOR PAPER
# Shows graphs on Kaggle screen and saves PNGs
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

SAVE_ROOT = "/kaggle/working/final_ieee_figures_clean"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ============================================================
# 1. FINAL VERIFIED CURRENT RESULTS
# ============================================================

unimodal = pd.DataFrame([
    {"Modality": "Fingerprint", "Domain": "Plain", "EER": 0.1685185185, "AUC": 0.9078251029, "Accuracy": 0.8312962963},
    {"Modality": "Fingerprint", "Domain": "True CKKS", "EER": 0.1685185185, "AUC": 0.9078251029, "Accuracy": 0.8312962963},
    {"Modality": "Face", "Domain": "Plain", "EER": 0.0566666667, "AUC": 0.9852945130, "Accuracy": 0.9437037037},
    {"Modality": "Face", "Domain": "True CKKS", "EER": 0.0566666667, "AUC": 0.9852945130, "Accuracy": 0.9437037037},
    {"Modality": "Iris", "Domain": "Plain", "EER": 0.0177777778, "AUC": 0.9979068587, "Accuracy": 0.9818518519},
    {"Modality": "Iris", "Domain": "True CKKS", "EER": 0.0177777778, "AUC": 0.9979068587, "Accuracy": 0.9818518519},
])

fusion = pd.DataFrame([
    {"System": "Face + Fingerprint", "EER": 0.0381481481, "AUC": 0.9932150892, "Accuracy": 0.9616666667},
    {"System": "Face + Iris", "EER": 0.0037037037, "AUC": 0.9999008230, "Accuracy": 0.9962962963},
    {"System": "Fingerprint + Iris", "EER": 0.0122222222, "AUC": 0.9992325103, "Accuracy": 0.9877777778},
    {"System": "All Three", "EER": 0.0029629630, "AUC": 0.9999507545, "Accuracy": 0.9972222222},
])

# ============================================================
# 2. OPTION 1 CORRECTED RESULTS
# ============================================================

ROOT = "/kaggle/working/design_space_revocability_unlinkability_fixed"

fp_df = pd.read_csv(os.path.join(ROOT, "fingerprint", "fingerprint_design_space_results.csv"))
face_df = pd.read_csv(os.path.join(ROOT, "face", "face_design_space_results.csv"))
iris_df = pd.read_csv(os.path.join(ROOT, "iris", "iris_design_space_results.csv"))

all_df = pd.concat([fp_df, face_df, iris_df], axis=0, ignore_index=True)

# ============================================================
# 3. HELPERS
# ============================================================

def load_scores(path, score_name):
    df = pd.read_csv(path)
    keep_cols = ["pair_id", "subject1", "subject2", "idx1", "idx2", "label", "score"]
    df = df[keep_cols].copy()
    return df.rename(columns={"score": score_name})

def normalize_scores(x):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def get_roc(df, score_col):
    y_true = df["label"].astype(int).values
    y_score = df[score_col].astype(float).values
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return fpr, tpr, auc

def fuse_all_three(face_path, fp_path, iris_path):
    face_df = load_scores(face_path, "score_face")
    fp_df   = load_scores(fp_path, "score_fingerprint")
    iris_df = load_scores(iris_path, "score_iris")

    merge_keys = ["pair_id", "subject1", "subject2", "idx1", "idx2", "label"]
    df = face_df.merge(fp_df, on=merge_keys).merge(iris_df, on=merge_keys)

    df["score_face_norm"] = normalize_scores(df["score_face"])
    df["score_fingerprint_norm"] = normalize_scores(df["score_fingerprint"])
    df["score_iris_norm"] = normalize_scores(df["score_iris"])

    df["fused_score"] = (
        0.30 * df["score_face_norm"] +
        0.20 * df["score_fingerprint_norm"] +
        0.50 * df["score_iris_norm"]
    )
    return df

def draw_annotated_heatmap(ax, matrix, row_labels, col_labels, title, value_fmt="{:.3f}"):
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title(title, fontsize=11)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, value_fmt.format(matrix[i, j]), ha="center", va="center", fontsize=9)

    return im

# ============================================================
# 4. FIGURE 1 - COMBINED ROC
# ============================================================

IRIS_PLAIN = "/kaggle/working/iris_cancellable_outputs_trueckks/iris_test_scores_cancellable_plain.csv"
IRIS_CKKS  = "/kaggle/working/iris_true_ckks_outputs/iris_test_scores_true_ckks.csv"

FP_PLAIN   = "/kaggle/working/fingerprint_cancellable_outputs_trueckks/fingerprint_test_scores_cancellable_plain.csv"
FACE_PLAIN = "/kaggle/working/face_cancellable_outputs_trueckks/face_test_scores_cancellable_plain.csv"
IRIS_PLAIN2= "/kaggle/working/iris_cancellable_outputs_trueckks/iris_test_scores_cancellable_plain.csv"

FP_CKKS    = "/kaggle/working/fingerprint_true_ckks_outputs/fingerprint_test_scores_true_ckks.csv"
FACE_CKKS  = "/kaggle/working/face_true_ckks_outputs/face_test_scores_true_ckks.csv"
IRIS_CKKS2 = "/kaggle/working/iris_true_ckks_outputs/iris_test_scores_true_ckks.csv"

iris_plain_df = pd.read_csv(IRIS_PLAIN)
iris_ckks_df  = pd.read_csv(IRIS_CKKS)

fusion_plain_df = fuse_all_three(FACE_PLAIN, FP_PLAIN, IRIS_PLAIN2)
fusion_ckks_df  = fuse_all_three(FACE_CKKS, FP_CKKS, IRIS_CKKS2)

fpr1, tpr1, auc1 = get_roc(iris_plain_df, "score")
fpr2, tpr2, auc2 = get_roc(iris_ckks_df, "score")
fpr3, tpr3, auc3 = get_roc(fusion_plain_df, "fused_score")
fpr4, tpr4, auc4 = get_roc(fusion_ckks_df, "fused_score")

plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, label=f"Iris Plain (AUC={auc1:.4f})")
plt.plot(fpr2, tpr2, label=f"Iris True CKKS (AUC={auc2:.4f})")
plt.plot(fpr3, tpr3, label=f"All-Three Plain Fusion (AUC={auc3:.4f})")
plt.plot(fpr4, tpr4, label=f"All-Three True CKKS Fusion (AUC={auc4:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Combined ROC: Best Unimodal and Best Fusion Systems")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig1_combined_roc.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 5. FIGURE 2 - SYSTEM RANKING BY EER
# ============================================================

rank_df = pd.DataFrame([
    {"System": "Fingerprint", "EER": 0.1685185185},
    {"System": "Face", "EER": 0.0566666667},
    {"System": "Iris", "EER": 0.0177777778},
    {"System": "Face + Fingerprint", "EER": 0.0381481481},
    {"System": "Face + Iris", "EER": 0.0037037037},
    {"System": "Fingerprint + Iris", "EER": 0.0122222222},
    {"System": "All Three", "EER": 0.0029629630},
]).sort_values("EER", ascending=True)

plt.figure(figsize=(9, 5.5))
plt.barh(rank_df["System"], rank_df["EER"])
plt.xlabel("EER")
plt.ylabel("System")
plt.title("System Ranking by EER (Lower is Better)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig2_system_ranking_eer.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 6. FIGURE 3 - PLAIN VS TRUE CKKS PARITY (UNIMODAL)
# ============================================================

mods = ["Fingerprint", "Face", "Iris"]
plain_eer = [unimodal[(unimodal["Modality"] == m) & (unimodal["Domain"] == "Plain")]["EER"].iloc[0] for m in mods]
ckks_eer  = [unimodal[(unimodal["Modality"] == m) & (unimodal["Domain"] == "True CKKS")]["EER"].iloc[0] for m in mods]

x = np.arange(len(mods))
w = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - w/2, plain_eer, width=w, label="Plain")
plt.bar(x + w/2, ckks_eer, width=w, label="True CKKS")
plt.xticks(x, mods)
plt.ylabel("EER")
plt.title("Plain vs True CKKS: Unimodal Protected Matching")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig3_unimodal_plain_vs_trueckks.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 7. FIGURE 4 - SAME-KEY EER HEATMAPS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, mod in zip(axes, ["fingerprint", "face", "iris"]):
    sub = all_df[(all_df["modality"] == mod) & (all_df["comparison_type"] == "same_key_mean")].copy()
    dims = [64, 128, 256, 512, 1024]
    binaries = [False, True]
    mat = []
    for d in dims:
        row = []
        for b in binaries:
            row.append(sub[(sub["dim"] == d) & (sub["binary"] == b)]["eer"].iloc[0])
        mat.append(row)
    mat = np.array(mat)
    im = draw_annotated_heatmap(ax, mat, dims, ["Non-binary", "Binary"], f"{mod.capitalize()}: Same-key EER")
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig4_samekey_eer_heatmaps.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 8. FIGURE 5 - CROSS-KEY EER HEATMAPS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, mod in zip(axes, ["fingerprint", "face", "iris"]):
    sub = all_df[(all_df["modality"] == mod) & (all_df["comparison_type"] == "cross_key_mean")].copy()
    dims = [64, 128, 256, 512, 1024]
    binaries = [False, True]
    mat = []
    for d in dims:
        row = []
        for b in binaries:
            row.append(sub[(sub["dim"] == d) & (sub["binary"] == b)]["eer"].iloc[0])
        mat.append(row)
    mat = np.array(mat)
    im = draw_annotated_heatmap(ax, mat, dims, ["Non-binary", "Binary"], f"{mod.capitalize()}: Cross-key EER")
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig5_crosskey_eer_heatmaps.png"), dpi=300, bbox_inches="tight")
plt.show()

print("All clean figures saved in:", SAVE_ROOT)

# ================= NOTEBOOK CELL 102 =================
# ============================================================
# CLEAN HEATMAPS WITHOUT OVERLAP
# Same-key EER + Cross-key EER
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SAVE_ROOT = "/kaggle/working/final_clean_heatmaps_no_overlap"
os.makedirs(SAVE_ROOT, exist_ok=True)

ROOT = "/kaggle/working/design_space_revocability_unlinkability_fixed"

fp_df = pd.read_csv(os.path.join(ROOT, "fingerprint", "fingerprint_design_space_results.csv"))
face_df = pd.read_csv(os.path.join(ROOT, "face", "face_design_space_results.csv"))
iris_df = pd.read_csv(os.path.join(ROOT, "iris", "iris_design_space_results.csv"))

all_df = pd.concat([fp_df, face_df, iris_df], axis=0, ignore_index=True)

def build_matrix(df, modality, comparison_type, metric):
    sub = df[(df["modality"] == modality) & (df["comparison_type"] == comparison_type)].copy()
    dims = [64, 128, 256, 512, 1024]
    binaries = [False, True]   # Non-binary, Binary

    mat = []
    for d in dims:
        row = []
        for b in binaries:
            val = sub[(sub["dim"] == d) & (sub["binary"] == b)][metric].iloc[0]
            row.append(val)
        mat.append(row)
    return np.array(mat)

def plot_heatmap_group(metric="eer", comparison_type="same_key_mean", title_prefix="Same-key EER", save_name="samekey_eer"):
    modalities = ["fingerprint", "face", "iris"]
    pretty_names = ["Fingerprint", "Face", "Iris"]
    dims = [64, 128, 256, 512, 1024]
    cols = ["Non-binary", "Binary"]

    mats = [build_matrix(all_df, m, comparison_type, metric) for m in modalities]

    # global scale same across all 3 heatmaps
    vmin = min(mat.min() for mat in mats)
    vmax = max(mat.max() for mat in mats)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)

    for ax, mat, name in zip(axes, mats, pretty_names):
        im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, fontsize=10)
        ax.set_yticks(np.arange(len(dims)))
        ax.set_yticklabels(dims, fontsize=10)
        ax.set_title(f"{name}: {title_prefix}", fontsize=11)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9)

    # shared colorbar outside all axes
    cbar = fig.colorbar(im, ax=axes, shrink=0.88, pad=0.03)
    cbar.ax.tick_params(labelsize=9)

    out_path = os.path.join(SAVE_ROOT, f"{save_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved:", out_path)

# ============================================================
# 1. SAME-KEY EER HEATMAPS
# ============================================================

plot_heatmap_group(
    metric="eer",
    comparison_type="same_key_mean",
    title_prefix="Same-key EER",
    save_name="samekey_eer_heatmaps_clean"
)

# ============================================================
# 2. CROSS-KEY EER HEATMAPS
# ============================================================

plot_heatmap_group(
    metric="eer",
    comparison_type="cross_key_mean",
    title_prefix="Cross-key EER",
    save_name="crosskey_eer_heatmaps_clean"
)

# ============================================================
# 3. CROSS-KEY ROC-AUC HEATMAPS
# ============================================================

plot_heatmap_group(
    metric="auc",
    comparison_type="cross_key_mean",
    title_prefix="Cross-key ROC-AUC",
    save_name="crosskey_auc_heatmaps_clean"
)

# ================= NOTEBOOK CELL 103 =================
# ============================================================
# RECOMMENDED CONFIGURATION SELECTION
# Best Utility / Best Unlinkability / Recommended Practical Config
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# ============================================================
# 1. PATHS
# ============================================================

ROOT = "/kaggle/working/design_space_revocability_unlinkability_fixed"
SAVE_ROOT = "/kaggle/working/recommended_config_analysis"
os.makedirs(SAVE_ROOT, exist_ok=True)

FP_PATH   = os.path.join(ROOT, "fingerprint", "fingerprint_design_space_results.csv")
FACE_PATH = os.path.join(ROOT, "face", "face_design_space_results.csv")
IRIS_PATH = os.path.join(ROOT, "iris", "iris_design_space_results.csv")

# ============================================================
# 2. LOAD
# ============================================================

fp_df   = pd.read_csv(FP_PATH)
face_df = pd.read_csv(FACE_PATH)
iris_df = pd.read_csv(IRIS_PATH)

all_df = pd.concat([fp_df, face_df, iris_df], axis=0, ignore_index=True)

# ============================================================
# 3. HELPERS
# ============================================================

def pretty_modality(x):
    return str(x).capitalize()

def choose_best_utility(df_mod):
    same_df = df_mod[df_mod["comparison_type"] == "same_key_mean"].copy()
    return same_df.loc[same_df["eer"].idxmin()]

def choose_best_unlinkability(df_mod):
    cross_df = df_mod[df_mod["comparison_type"] == "cross_key_mean"].copy()
    cross_df["auc_dist_from_0_5"] = np.abs(cross_df["auc"] - 0.5)
    return cross_df.loc[cross_df["auc_dist_from_0_5"].idxmin()]

def choose_recommended(df_mod):
    same_df = df_mod[df_mod["comparison_type"] == "same_key_mean"].copy()
    cross_df = df_mod[df_mod["comparison_type"] == "cross_key_mean"].copy()

    merged = same_df.merge(
        cross_df,
        on=["modality", "dim", "binary"],
        suffixes=("_same", "_cross")
    )

    # Normalize metrics for balanced selection
    merged = merged.copy()

    # lower same-key EER is better
    merged["same_eer_norm"] = merged["eer_same"] / (merged["eer_same"].max() + 1e-12)

    # cross-key AUC should be close to 0.5
    merged["cross_auc_dist"] = np.abs(merged["auc_cross"] - 0.5)
    merged["cross_auc_dist_norm"] = merged["cross_auc_dist"] / (merged["cross_auc_dist"].max() + 1e-12)

    # cross-key similarity should be close to 0 for non-binary and generally low
    merged["cross_sim_abs"] = np.abs(merged["mean_cross_key_similarity_cross"])
    merged["cross_sim_abs_norm"] = merged["cross_sim_abs"] / (merged["cross_sim_abs"].max() + 1e-12)

    # Weighted composite score
    # utility most important, then AUC closeness, then similarity
    merged["recommended_score"] = (
        0.50 * merged["same_eer_norm"] +
        0.30 * merged["cross_auc_dist_norm"] +
        0.20 * merged["cross_sim_abs_norm"]
    )

    return merged.loc[merged["recommended_score"].idxmin()]

def format_table(df):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{x:.4f}")
    return out

# ============================================================
# 4. BUILD TABLES
# ============================================================

rows_summary = []
rows_detailed = []

for mod in ["fingerprint", "face", "iris"]:
    df_mod = all_df[all_df["modality"] == mod].copy()

    best_u = choose_best_utility(df_mod)
    best_x = choose_best_unlinkability(df_mod)
    rec    = choose_recommended(df_mod)

    rows_summary.append({
        "Modality": pretty_modality(mod),

        "Best Utility Dim": int(best_u["dim"]),
        "Best Utility Binary": bool(best_u["binary"]),
        "Best Utility EER": float(best_u["eer"]),
        "Best Utility AUC": float(best_u["auc"]),
        "Best Utility Accuracy": float(best_u["accuracy"]),

        "Best Unlinkability Dim": int(best_x["dim"]),
        "Best Unlinkability Binary": bool(best_x["binary"]),
        "Best Unlinkability Cross-key EER": float(best_x["eer"]),
        "Best Unlinkability Cross-key AUC": float(best_x["auc"]),
        "Best Unlinkability Mean Similarity": float(best_x["mean_cross_key_similarity"]),

        "Recommended Dim": int(rec["dim"]),
        "Recommended Binary": bool(rec["binary"]),
        "Recommended Same-key EER": float(rec["eer_same"]),
        "Recommended Same-key AUC": float(rec["auc_same"]),
        "Recommended Same-key Accuracy": float(rec["accuracy_same"]),
        "Recommended Cross-key EER": float(rec["eer_cross"]),
        "Recommended Cross-key AUC": float(rec["auc_cross"]),
        "Recommended Mean Cross-key Similarity": float(rec["mean_cross_key_similarity_cross"]),
        "Recommended Composite Score": float(rec["recommended_score"]),
    })

    rows_detailed.append({
        "Modality": pretty_modality(mod),
        "Selection": "Best Utility",
        "Dim": int(best_u["dim"]),
        "Binary": bool(best_u["binary"]),
        "Same-key EER": float(best_u["eer"]),
        "Same-key AUC": float(best_u["auc"]),
        "Same-key Accuracy": float(best_u["accuracy"]),
        "Cross-key EER": np.nan,
        "Cross-key AUC": np.nan,
        "Mean Cross-key Similarity": np.nan,
    })

    rows_detailed.append({
        "Modality": pretty_modality(mod),
        "Selection": "Best Unlinkability",
        "Dim": int(best_x["dim"]),
        "Binary": bool(best_x["binary"]),
        "Same-key EER": np.nan,
        "Same-key AUC": np.nan,
        "Same-key Accuracy": np.nan,
        "Cross-key EER": float(best_x["eer"]),
        "Cross-key AUC": float(best_x["auc"]),
        "Mean Cross-key Similarity": float(best_x["mean_cross_key_similarity"]),
    })

    rows_detailed.append({
        "Modality": pretty_modality(mod),
        "Selection": "Recommended",
        "Dim": int(rec["dim"]),
        "Binary": bool(rec["binary"]),
        "Same-key EER": float(rec["eer_same"]),
        "Same-key AUC": float(rec["auc_same"]),
        "Same-key Accuracy": float(rec["accuracy_same"]),
        "Cross-key EER": float(rec["eer_cross"]),
        "Cross-key AUC": float(rec["auc_cross"]),
        "Mean Cross-key Similarity": float(rec["mean_cross_key_similarity_cross"]),
    })

summary_df = pd.DataFrame(rows_summary)
detailed_df = pd.DataFrame(rows_detailed)

# ============================================================
# 5. DISPLAY TABLES
# ============================================================

print("\n" + "="*110)
print("TABLE 1. BEST UTILITY, BEST UNLINKABILITY, AND RECOMMENDED CONFIGURATION")
print("="*110)
display(format_table(summary_df))

print("\n" + "="*110)
print("TABLE 2. DETAILED SELECTION VIEW")
print("="*110)
display(format_table(detailed_df))

# ============================================================
# 6. SAVE TABLES
# ============================================================

summary_df.to_csv(os.path.join(SAVE_ROOT, "recommended_config_summary.csv"), index=False)
detailed_df.to_csv(os.path.join(SAVE_ROOT, "recommended_config_detailed.csv"), index=False)

format_table(summary_df).to_csv(os.path.join(SAVE_ROOT, "recommended_config_summary_display.csv"), index=False)
format_table(detailed_df).to_csv(os.path.join(SAVE_ROOT, "recommended_config_detailed_display.csv"), index=False)

# ============================================================
# 7. CLEAN SUMMARY CHARTS
# ============================================================

# ----- Chart 1: Recommended Same-key EER -----
plt.figure(figsize=(8, 5))
plt.bar(summary_df["Modality"], summary_df["Recommended Same-key EER"])
plt.ylabel("Recommended Same-key EER")
plt.title("Recommended Protected Configuration per Modality")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig1_recommended_samekey_eer.png"), dpi=300, bbox_inches="tight")
plt.show()

# ----- Chart 2: Recommended Cross-key AUC -----
plt.figure(figsize=(8, 5))
plt.bar(summary_df["Modality"], summary_df["Recommended Cross-key AUC"])
plt.ylabel("Recommended Cross-key ROC-AUC")
plt.title("Recommended Configuration: Cross-key Unlinkability")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig2_recommended_crosskey_auc.png"), dpi=300, bbox_inches="tight")
plt.show()

# ----- Chart 3: Dim comparison across selections -----
plot_df = detailed_df.copy()
plot_df["Label"] = plot_df["Modality"] + " - " + plot_df["Selection"]

plt.figure(figsize=(10, 5))
plt.bar(plot_df["Label"], plot_df["Dim"])
plt.xticks(rotation=25, ha="right")
plt.ylabel("Projection Dimension")
plt.title("Selected Protected Configuration Dimensions")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig3_selected_dimensions.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 8. SAVE TEXT SUMMARY
# ============================================================

with open(os.path.join(SAVE_ROOT, "recommended_config_summary.txt"), "w") as f:
    f.write("Recommended Configuration Analysis\n\n")
    for _, row in summary_df.iterrows():
        f.write(f"Modality: {row['Modality']}\n")
        f.write(f"  Best Utility      -> Dim={row['Best Utility Dim']}, Binary={row['Best Utility Binary']}, EER={row['Best Utility EER']:.4f}\n")
        f.write(f"  Best Unlinkability-> Dim={row['Best Unlinkability Dim']}, Binary={row['Best Unlinkability Binary']}, Cross-key AUC={row['Best Unlinkability Cross-key AUC']:.4f}\n")
        f.write(f"  Recommended       -> Dim={row['Recommended Dim']}, Binary={row['Recommended Binary']}, Same-key EER={row['Recommended Same-key EER']:.4f}, Cross-key AUC={row['Recommended Cross-key AUC']:.4f}\n\n")

print("\nAll files saved in:", SAVE_ROOT)

# ================= NOTEBOOK CELL 104 =================
# ============================================================
# FINGERPRINT RECOMMENDED CONFIG: TRUE CKKS VALIDATION
# Recommended: dim=1024, binary=False
# ============================================================

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tenseal as ts

EMB_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/fingerprint_balanced_test_outputs/fingerprint_test_embeddings.npy"
META_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/fingerprint_balanced_test_outputs/fingerprint_test_embeddings_meta.csv"
PAIRS_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/common_pairs_balanced/test_pairs_common_balanced.csv"

OUT_CANC_DIR = "/kaggle/working/fingerprint_recommended_outputs"
OUT_CKKS_DIR = "/kaggle/working/fingerprint_recommended_trueckks_outputs"
os.makedirs(OUT_CANC_DIR, exist_ok=True)
os.makedirs(OUT_CKKS_DIR, exist_ok=True)

KEY = 11
OUT_DIM = 1024
BINARY = False

poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
global_scale = 2**40

embeddings = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH)
pairs = pd.read_csv(PAIRS_PATH)

meta = meta.copy()
meta["global_index"] = np.arange(len(meta))
meta = meta.sort_values(["subject", "image_path"]).reset_index(drop=True)
meta["local_idx"] = meta.groupby("subject").cumcount()

index_map = {(int(r.subject), int(r.local_idx)): int(r.global_index) for _, r in meta.iterrows()}

def cancellable_transform(emb, key=11, out_dim=1024, binary=False):
    rng = np.random.default_rng(seed=key)
    R = rng.standard_normal((emb.shape[1], out_dim))
    proj = emb @ R
    proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
    if binary:
        return (proj > 0).astype(np.float32)
    return proj.astype(np.float32)

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
        "eer": eer, "threshold": thr, "auc": auc_val,
        "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
        "cm": cm
    }

c_embeddings = cancellable_transform(embeddings, key=KEY, out_dim=OUT_DIM, binary=BINARY)
np.save(os.path.join(OUT_CANC_DIR, "fingerprint_recommended_embeddings.npy"), c_embeddings)

# Plain
plain_scores = []
for _, row in pairs.iterrows():
    g1 = index_map[(int(row["subject1"]), int(row["idx1"]))]
    g2 = index_map[(int(row["subject2"]), int(row["idx2"]))]
    score = float(np.dot(c_embeddings[g1], c_embeddings[g2]))
    plain_scores.append(score)

plain_df = pairs.copy()
plain_df["score"] = plain_scores
plain_df.to_csv(os.path.join(OUT_CANC_DIR, "fingerprint_recommended_plain_scores.csv"), index=False)

plain_metrics = compute_metrics(plain_df["label"].astype(int).values, plain_df["score"].astype(float).values)

# CKKS
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly_modulus_degree, coeff_mod_bit_sizes=coeff_mod_bit_sizes)
context.generate_galois_keys()
context.global_scale = global_scale

t0 = time.time()
enc_gallery = [ts.ckks_vector(context, vec.tolist()) for vec in c_embeddings]
encrypt_time = time.time() - t0

t1 = time.time()
ckks_scores = []
for _, row in pairs.iterrows():
    probe_idx = index_map[(int(row["subject1"]), int(row["idx1"]))]
    gallery_idx = index_map[(int(row["subject2"]), int(row["idx2"]))]
    probe_vec = c_embeddings[probe_idx].astype(np.float32)
    enc_score = enc_gallery[gallery_idx].dot(probe_vec.tolist())
    dec_score = enc_score.decrypt()
    score = float(dec_score[0]) if isinstance(dec_score, (list, tuple, np.ndarray)) else float(dec_score)
    ckks_scores.append(score)
score_time = time.time() - t1

ckks_df = pairs.copy()
ckks_df["score"] = ckks_scores
ckks_df.to_csv(os.path.join(OUT_CKKS_DIR, "fingerprint_recommended_trueckks_scores.csv"), index=False)

ckks_metrics = compute_metrics(ckks_df["label"].astype(int).values, ckks_df["score"].astype(float).values)

compare_df = plain_df[["pair_id", "score"]].rename(columns={"score": "plain_score"}).merge(
    ckks_df[["pair_id", "score"]].rename(columns={"score": "ckks_score"}),
    on="pair_id"
)
compare_df["abs_diff"] = np.abs(compare_df["plain_score"] - compare_df["ckks_score"])
compare_df.to_csv(os.path.join(OUT_CKKS_DIR, "fingerprint_recommended_plain_vs_ckks.csv"), index=False)

final_metrics = {
    "modality": "fingerprint",
    "recommended_dim": OUT_DIM,
    "recommended_binary": BINARY,
    "plain_eer": plain_metrics["eer"],
    "plain_auc": plain_metrics["auc"],
    "plain_accuracy": plain_metrics["accuracy"],
    "ckks_eer": ckks_metrics["eer"],
    "ckks_auc": ckks_metrics["auc"],
    "ckks_accuracy": ckks_metrics["accuracy"],
    "mean_abs_diff_vs_plain": float(compare_df["abs_diff"].mean()),
    "max_abs_diff_vs_plain": float(compare_df["abs_diff"].max()),
    "encryption_time_sec": float(encrypt_time),
    "pair_scoring_time_sec": float(score_time),
    "total_time_sec": float(encrypt_time + score_time),
}
with open(os.path.join(OUT_CKKS_DIR, "fingerprint_recommended_metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=4)

print("===== FINGERPRINT RECOMMENDED CONFIG =====")
print(final_metrics)
print("Saved in:", OUT_CKKS_DIR)

# ================= NOTEBOOK CELL 105 =================
# ============================================================
# FACE RECOMMENDED CONFIG: TRUE CKKS VALIDATION
# Recommended: dim=256, binary=False
# ============================================================

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tenseal as ts

EMB_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/face_pretrained_embeddings/face_test_embeddings_pretrained.npy"
META_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/face_pretrained_embeddings/face_test_embeddings_meta_pretrained.csv"
PAIRS_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/common_pairs_balanced/test_pairs_common_balanced.csv"

OUT_CANC_DIR = "/kaggle/working/face_recommended_outputs"
OUT_CKKS_DIR = "/kaggle/working/face_recommended_trueckks_outputs"
os.makedirs(OUT_CANC_DIR, exist_ok=True)
os.makedirs(OUT_CKKS_DIR, exist_ok=True)

KEY = 11
OUT_DIM = 256
BINARY = False

poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
global_scale = 2**40

embeddings = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH)
pairs = pd.read_csv(PAIRS_PATH)

meta = meta.copy()
meta["global_index"] = np.arange(len(meta))
meta = meta.sort_values(["subject", "image_path"]).reset_index(drop=True)
meta["local_idx"] = meta.groupby("subject").cumcount()
index_map = {(int(r.subject), int(r.local_idx)): int(r.global_index) for _, r in meta.iterrows()}

def cancellable_transform(emb, key=11, out_dim=256, binary=False):
    rng = np.random.default_rng(seed=key)
    R = rng.standard_normal((emb.shape[1], out_dim))
    proj = emb @ R
    proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
    if binary:
        return (proj > 0).astype(np.float32)
    return proj.astype(np.float32)

def compute_metrics(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(fpr[eer_idx]); thr = float(thresholds[eer_idx])
    if not np.isfinite(thr): thr = 0.0
    auc_val = float(roc_auc_score(y_true, y_score))
    y_pred = (y_score >= thr).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    return {"eer": eer, "threshold": thr, "auc": auc_val, "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "cm": cm}

c_embeddings = cancellable_transform(embeddings, key=KEY, out_dim=OUT_DIM, binary=BINARY)
np.save(os.path.join(OUT_CANC_DIR, "face_recommended_embeddings.npy"), c_embeddings)

plain_scores = []
for _, row in pairs.iterrows():
    g1 = index_map[(int(row["subject1"]), int(row["idx1"]))]
    g2 = index_map[(int(row["subject2"]), int(row["idx2"]))]
    plain_scores.append(float(np.dot(c_embeddings[g1], c_embeddings[g2])))

plain_df = pairs.copy()
plain_df["score"] = plain_scores
plain_df.to_csv(os.path.join(OUT_CANC_DIR, "face_recommended_plain_scores.csv"), index=False)
plain_metrics = compute_metrics(plain_df["label"].astype(int).values, plain_df["score"].astype(float).values)

context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly_modulus_degree, coeff_mod_bit_sizes=coeff_mod_bit_sizes)
context.generate_galois_keys()
context.global_scale = global_scale

t0 = time.time()
enc_gallery = [ts.ckks_vector(context, vec.tolist()) for vec in c_embeddings]
encrypt_time = time.time() - t0

t1 = time.time()
ckks_scores = []
for _, row in pairs.iterrows():
    probe_idx = index_map[(int(row["subject1"]), int(row["idx1"]))]
    gallery_idx = index_map[(int(row["subject2"]), int(row["idx2"]))]
    probe_vec = c_embeddings[probe_idx].astype(np.float32)
    enc_score = enc_gallery[gallery_idx].dot(probe_vec.tolist())
    dec_score = enc_score.decrypt()
    score = float(dec_score[0]) if isinstance(dec_score, (list, tuple, np.ndarray)) else float(dec_score)
    ckks_scores.append(score)
score_time = time.time() - t1

ckks_df = pairs.copy()
ckks_df["score"] = ckks_scores
ckks_df.to_csv(os.path.join(OUT_CKKS_DIR, "face_recommended_trueckks_scores.csv"), index=False)
ckks_metrics = compute_metrics(ckks_df["label"].astype(int).values, ckks_df["score"].astype(float).values)

compare_df = plain_df[["pair_id", "score"]].rename(columns={"score": "plain_score"}).merge(
    ckks_df[["pair_id", "score"]].rename(columns={"score": "ckks_score"}), on="pair_id"
)
compare_df["abs_diff"] = np.abs(compare_df["plain_score"] - compare_df["ckks_score"])
compare_df.to_csv(os.path.join(OUT_CKKS_DIR, "face_recommended_plain_vs_ckks.csv"), index=False)

final_metrics = {
    "modality": "face",
    "recommended_dim": OUT_DIM,
    "recommended_binary": BINARY,
    "plain_eer": plain_metrics["eer"],
    "plain_auc": plain_metrics["auc"],
    "plain_accuracy": plain_metrics["accuracy"],
    "ckks_eer": ckks_metrics["eer"],
    "ckks_auc": ckks_metrics["auc"],
    "ckks_accuracy": ckks_metrics["accuracy"],
    "mean_abs_diff_vs_plain": float(compare_df["abs_diff"].mean()),
    "max_abs_diff_vs_plain": float(compare_df["abs_diff"].max()),
    "encryption_time_sec": float(encrypt_time),
    "pair_scoring_time_sec": float(score_time),
    "total_time_sec": float(encrypt_time + score_time),
}
with open(os.path.join(OUT_CKKS_DIR, "face_recommended_metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=4)

print("===== FACE RECOMMENDED CONFIG =====")
print(final_metrics)
print("Saved in:", OUT_CKKS_DIR)

# ================= NOTEBOOK CELL 106 =================
# ============================================================
# IRIS RECOMMENDED CONFIG: TRUE CKKS VALIDATION
# Recommended: dim=1024, binary=False
# ============================================================

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tenseal as ts

EMB_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/iris_balanced_test_outputs/iris_test_embeddings.npy"
META_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/iris_balanced_test_outputs/iris_test_embeddings_meta.csv"
PAIRS_PATH = "/kaggle/input/datasets/radhe11/backup/kaggle/working/common_pairs_balanced/test_pairs_common_balanced.csv"

OUT_CANC_DIR = "/kaggle/working/iris_recommended_outputs"
OUT_CKKS_DIR = "/kaggle/working/iris_recommended_trueckks_outputs"
os.makedirs(OUT_CANC_DIR, exist_ok=True)
os.makedirs(OUT_CKKS_DIR, exist_ok=True)

KEY = 11
OUT_DIM = 1024
BINARY = False

poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
global_scale = 2**40

embeddings = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH)
pairs = pd.read_csv(PAIRS_PATH)

meta = meta.copy()
meta["global_index"] = np.arange(len(meta))
meta = meta.sort_values(["subject", "image_path"]).reset_index(drop=True)
meta["local_idx"] = meta.groupby("subject").cumcount()
index_map = {(int(r.subject), int(r.local_idx)): int(r.global_index) for _, r in meta.iterrows()}

def cancellable_transform(emb, key=11, out_dim=1024, binary=False):
    rng = np.random.default_rng(seed=key)
    R = rng.standard_normal((emb.shape[1], out_dim))
    proj = emb @ R
    proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
    if binary:
        return (proj > 0).astype(np.float32)
    return proj.astype(np.float32)

def compute_metrics(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(fpr[eer_idx]); thr = float(thresholds[eer_idx])
    if not np.isfinite(thr): thr = 0.0
    auc_val = float(roc_auc_score(y_true, y_score))
    y_pred = (y_score >= thr).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    return {"eer": eer, "threshold": thr, "auc": auc_val, "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "cm": cm}

c_embeddings = cancellable_transform(embeddings, key=KEY, out_dim=OUT_DIM, binary=BINARY)
np.save(os.path.join(OUT_CANC_DIR, "iris_recommended_embeddings.npy"), c_embeddings)

plain_scores = []
for _, row in pairs.iterrows():
    g1 = index_map[(int(row["subject1"]), int(row["idx1"]))]
    g2 = index_map[(int(row["subject2"]), int(row["idx2"]))]
    plain_scores.append(float(np.dot(c_embeddings[g1], c_embeddings[g2])))

plain_df = pairs.copy()
plain_df["score"] = plain_scores
plain_df.to_csv(os.path.join(OUT_CANC_DIR, "iris_recommended_plain_scores.csv"), index=False)
plain_metrics = compute_metrics(plain_df["label"].astype(int).values, plain_df["score"].astype(float).values)

context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly_modulus_degree, coeff_mod_bit_sizes=coeff_mod_bit_sizes)
context.generate_galois_keys()
context.global_scale = global_scale

t0 = time.time()
enc_gallery = [ts.ckks_vector(context, vec.tolist()) for vec in c_embeddings]
encrypt_time = time.time() - t0

t1 = time.time()
ckks_scores = []
for _, row in pairs.iterrows():
    probe_idx = index_map[(int(row["subject1"]), int(row["idx1"]))]
    gallery_idx = index_map[(int(row["subject2"]), int(row["idx2"]))]
    probe_vec = c_embeddings[probe_idx].astype(np.float32)
    enc_score = enc_gallery[gallery_idx].dot(probe_vec.tolist())
    dec_score = enc_score.decrypt()
    score = float(dec_score[0]) if isinstance(dec_score, (list, tuple, np.ndarray)) else float(dec_score)
    ckks_scores.append(score)
score_time = time.time() - t1

ckks_df = pairs.copy()
ckks_df["score"] = ckks_scores
ckks_df.to_csv(os.path.join(OUT_CKKS_DIR, "iris_recommended_trueckks_scores.csv"), index=False)
ckks_metrics = compute_metrics(ckks_df["label"].astype(int).values, ckks_df["score"].astype(float).values)

compare_df = plain_df[["pair_id", "score"]].rename(columns={"score": "plain_score"}).merge(
    ckks_df[["pair_id", "score"]].rename(columns={"score": "ckks_score"}), on="pair_id"
)
compare_df["abs_diff"] = np.abs(compare_df["plain_score"] - compare_df["ckks_score"])
compare_df.to_csv(os.path.join(OUT_CKKS_DIR, "iris_recommended_plain_vs_ckks.csv"), index=False)

final_metrics = {
    "modality": "iris",
    "recommended_dim": OUT_DIM,
    "recommended_binary": BINARY,
    "plain_eer": plain_metrics["eer"],
    "plain_auc": plain_metrics["auc"],
    "plain_accuracy": plain_metrics["accuracy"],
    "ckks_eer": ckks_metrics["eer"],
    "ckks_auc": ckks_metrics["auc"],
    "ckks_accuracy": ckks_metrics["accuracy"],
    "mean_abs_diff_vs_plain": float(compare_df["abs_diff"].mean()),
    "max_abs_diff_vs_plain": float(compare_df["abs_diff"].max()),
    "encryption_time_sec": float(encrypt_time),
    "pair_scoring_time_sec": float(score_time),
    "total_time_sec": float(encrypt_time + score_time),
}
with open(os.path.join(OUT_CKKS_DIR, "iris_recommended_metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=4)

print("===== IRIS RECOMMENDED CONFIG =====")
print(final_metrics)
print("Saved in:", OUT_CKKS_DIR)

# ================= NOTEBOOK CELL 107 =================
# ============================================================
# RECOMMENDED-CONFIG FUSION
# Plain Fusion + True CKKS Fusion + Graphs
# Recommended configs:
#   Fingerprint -> 1024, non-binary
#   Face        -> 256,  non-binary
#   Iris        -> 1024, non-binary
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

SAVE_ROOT = "/kaggle/working/recommended_config_fusion_outputs"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ------------------------------------------------------------
# 1. INPUT PATHS
# ------------------------------------------------------------

# Plain recommended scores
FP_PLAIN_PATH   = "/kaggle/working/fingerprint_recommended_outputs/fingerprint_recommended_plain_scores.csv"
FACE_PLAIN_PATH = "/kaggle/working/face_recommended_outputs/face_recommended_plain_scores.csv"
IRIS_PLAIN_PATH = "/kaggle/working/iris_recommended_outputs/iris_recommended_plain_scores.csv"

# True CKKS recommended scores
FP_CKKS_PATH    = "/kaggle/working/fingerprint_recommended_trueckks_outputs/fingerprint_recommended_trueckks_scores.csv"
FACE_CKKS_PATH  = "/kaggle/working/face_recommended_trueckks_outputs/face_recommended_trueckks_scores.csv"
IRIS_CKKS_PATH  = "/kaggle/working/iris_recommended_trueckks_outputs/iris_recommended_trueckks_scores.csv"

# ------------------------------------------------------------
# 2. FUSION WEIGHTS
# Reuse validated fusion weights from earlier work
# ------------------------------------------------------------

PAIR_WEIGHTS = {
    "face_fingerprint": {"face": 0.60, "fingerprint": 0.40},
    "face_iris": {"face": 0.375, "iris": 0.625},
    "fingerprint_iris": {"fingerprint": 0.2857, "iris": 0.7143},
    "all_three": {"face": 0.30, "fingerprint": 0.20, "iris": 0.50},
}

fusion_sets = {
    "face_fingerprint": ["face", "fingerprint"],
    "face_iris": ["face", "iris"],
    "fingerprint_iris": ["fingerprint", "iris"],
    "all_three": ["face", "fingerprint", "iris"]
}

# ------------------------------------------------------------
# 3. HELPERS
# ------------------------------------------------------------

def load_score_file(path, modality_name):
    df = pd.read_csv(path)
    keep_cols = ["pair_id", "subject1", "subject2", "idx1", "idx2", "label", "score"]
    df = df[keep_cols].copy()
    df = df.rename(columns={"score": f"score_{modality_name}"})
    return df

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

def save_plots(df, metrics, out_dir, title_prefix):
    os.makedirs(out_dir, exist_ok=True)

    fpr = metrics["fpr"]
    tpr = metrics["tpr"]
    thr = metrics["threshold"]
    auc_val = metrics["auc"]
    cm = metrics["cm"]

    # ROC
    plt.figure(figsize=(6.5, 5))
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

# ------------------------------------------------------------
# 4. LOAD PLAIN + CKKS SCORES
# ------------------------------------------------------------

merge_keys = ["pair_id", "subject1", "subject2", "idx1", "idx2", "label"]

plain_face = load_score_file(FACE_PLAIN_PATH, "face")
plain_fp   = load_score_file(FP_PLAIN_PATH, "fingerprint")
plain_iris = load_score_file(IRIS_PLAIN_PATH, "iris")

ckks_face = load_score_file(FACE_CKKS_PATH, "face")
ckks_fp   = load_score_file(FP_CKKS_PATH, "fingerprint")
ckks_iris = load_score_file(IRIS_CKKS_PATH, "iris")

plain_all = plain_face.merge(plain_fp, on=merge_keys).merge(plain_iris, on=merge_keys)
ckks_all  = ckks_face.merge(ckks_fp, on=merge_keys).merge(ckks_iris, on=merge_keys)

# normalize scores per modality
for df in [plain_all, ckks_all]:
    df["score_face_norm"] = normalize_scores(df["score_face"])
    df["score_fingerprint_norm"] = normalize_scores(df["score_fingerprint"])
    df["score_iris_norm"] = normalize_scores(df["score_iris"])

print("Plain merged preview:")
print(plain_all.head())

print("\nTrue CKKS merged preview:")
print(ckks_all.head())

# ------------------------------------------------------------
# 5. RUN FUSIONS FOR BOTH DOMAINS
# ------------------------------------------------------------

summary_rows = []

for domain_name, df_source in [("recommended_plain", plain_all), ("recommended_true_ckks", ckks_all)]:
    for fusion_name, mods in fusion_sets.items():
        df = df_source.copy()

        weight_dict = PAIR_WEIGHTS[fusion_name]
        weights = np.array([weight_dict[m] for m in mods], dtype=np.float32)
        weights = weights / weights.sum()

        score_cols = [f"score_{m}_norm" for m in mods]
        score_mat = df[score_cols].values.astype(np.float32)

        df["fused_score"] = np.sum(score_mat * weights.reshape(1, -1), axis=1)

        y_true = df["label"].values.astype(int)
        metrics = compute_metrics(y_true, df["fused_score"].values)

        out_dir = os.path.join(SAVE_ROOT, domain_name, fusion_name)
        os.makedirs(out_dir, exist_ok=True)

        df.to_csv(os.path.join(out_dir, f"{fusion_name}_scores.csv"), index=False)

        summary = {
            "mode": domain_name,
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

        pd.DataFrame([summary]).to_csv(os.path.join(out_dir, f"{fusion_name}_result_row.csv"), index=False)

        save_plots(df, metrics, out_dir, f"{domain_name}: {fusion_name}")

        summary_rows.append(summary)

        print(f"\nDone: {domain_name} -> {fusion_name}")
        print(summary)

# ------------------------------------------------------------
# 6. FINAL SUMMARY TABLE
# ------------------------------------------------------------

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(SAVE_ROOT, "recommended_config_fusion_summary.csv")
summary_df.to_csv(summary_csv, index=False)

print("\n===== FINAL RECOMMENDED-CONFIG FUSION SUMMARY =====")
print(summary_df)
print("\nSaved:", summary_csv)

# ------------------------------------------------------------
# 7. CLEAN COMPARISON CHARTS (PLAIN VS CKKS)
# ------------------------------------------------------------

name_map = {
    "face_fingerprint": "Face + Fingerprint",
    "face_iris": "Face + Iris",
    "fingerprint_iris": "Fingerprint + Iris",
    "all_three": "All Three"
}

plain_sum = summary_df[summary_df["mode"] == "recommended_plain"].copy()
ckks_sum  = summary_df[summary_df["mode"] == "recommended_true_ckks"].copy()

plain_sum["Fusion"] = plain_sum["fusion_name"].map(name_map)
ckks_sum["Fusion"] = ckks_sum["fusion_name"].map(name_map)

plain_sum = plain_sum.sort_values("fusion_name").reset_index(drop=True)
ckks_sum = ckks_sum.sort_values("fusion_name").reset_index(drop=True)

x = np.arange(len(plain_sum))
w = 0.35

# EER comparison
plt.figure(figsize=(9, 5))
plt.bar(x - w/2, plain_sum["eer"], width=w, label="Recommended Plain")
plt.bar(x + w/2, ckks_sum["eer"], width=w, label="Recommended True CKKS")
plt.xticks(x, plain_sum["Fusion"], rotation=20, ha="right")
plt.ylabel("EER")
plt.title("Recommended-Config Fusion: Plain vs True CKKS")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig_plain_vs_ckks_recommended_fusion_eer.png"), dpi=300, bbox_inches="tight")
plt.show()

# Accuracy comparison
plt.figure(figsize=(9, 5))
plt.bar(x - w/2, plain_sum["accuracy"], width=w, label="Recommended Plain")
plt.bar(x + w/2, ckks_sum["accuracy"], width=w, label="Recommended True CKKS")
plt.xticks(x, plain_sum["Fusion"], rotation=20, ha="right")
plt.ylabel("Accuracy")
plt.title("Recommended-Config Fusion Accuracy: Plain vs True CKKS")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig_plain_vs_ckks_recommended_fusion_accuracy.png"), dpi=300, bbox_inches="tight")
plt.show()

print("\nAll outputs saved in:", SAVE_ROOT)

# ================= NOTEBOOK CELL 114 =================
# ============================================================
# FINAL MASTER TABLE + CLEAN PAPER GRAPHS
# Combines:
# 1) Unimodal plain vs true CKKS
# 2) Recommended-config unimodal true CKKS
# 3) Recommended-config fusion plain vs true CKKS
# 4) Binary vs non-binary tradeoff
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

SAVE_ROOT = "/kaggle/working/final_master_paper_results"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ============================================================
# 1. INPUT PATHS
# ============================================================

# original fixed-config true CKKS metrics
FP_CKKS_JSON   = "/kaggle/working/fingerprint_true_ckks_outputs/fingerprint_true_ckks_metrics.json"
FACE_CKKS_JSON = "/kaggle/working/face_true_ckks_outputs/face_true_ckks_metrics.json"
IRIS_CKKS_JSON = "/kaggle/working/iris_true_ckks_outputs/iris_true_ckks_metrics.json"

# recommended-config CKKS metrics
FP_REC_JSON   = "/kaggle/working/fingerprint_recommended_trueckks_outputs/fingerprint_recommended_metrics.json"
FACE_REC_JSON = "/kaggle/working/face_recommended_trueckks_outputs/face_recommended_metrics.json"
IRIS_REC_JSON = "/kaggle/working/iris_recommended_trueckks_outputs/iris_recommended_metrics.json"

# recommended-config fusion summary
REC_FUSION_CSV = "/kaggle/working/recommended_config_fusion_outputs/recommended_config_fusion_summary.csv"

# binary vs non-binary irreversibility comparison
FP_BIN_CSV   = "/kaggle/working/binary_vs_nonbinary_irreversibility_fingerprint/fingerprint_binary_vs_nonbinary_irreversibility.csv"
FACE_BIN_CSV = "/kaggle/working/binary_vs_nonbinary_irreversibility_face/face_binary_vs_nonbinary_irreversibility.csv"
IRIS_BIN_CSV = "/kaggle/working/binary_vs_nonbinary_irreversibility_iris/iris_binary_vs_nonbinary_irreversibility.csv"

# ============================================================
# 2. HELPERS
# ============================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def round_df(df, sci_cols=None):
    out = df.copy()
    sci_cols = sci_cols or []
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if col in sci_cols:
                out[col] = out[col].map(lambda x: f"{x:.4e}")
            else:
                out[col] = out[col].map(lambda x: f"{x:.4f}")
    return out

# ============================================================
# 3. LOAD DATA
# ============================================================

fp_ckks = load_json(FP_CKKS_JSON)
face_ckks = load_json(FACE_CKKS_JSON)
iris_ckks = load_json(IRIS_CKKS_JSON)

fp_rec = load_json(FP_REC_JSON)
face_rec = load_json(FACE_REC_JSON)
iris_rec = load_json(IRIS_REC_JSON)

rec_fusion = pd.read_csv(REC_FUSION_CSV)

fp_bin = pd.read_csv(FP_BIN_CSV)
face_bin = pd.read_csv(FACE_BIN_CSV)
iris_bin = pd.read_csv(IRIS_BIN_CSV)

bin_all = pd.concat([fp_bin, face_bin, iris_bin], axis=0, ignore_index=True)

# ============================================================
# 4. TABLE 1 - UNIMODAL PLAIN VS TRUE CKKS (CURRENT FINAL)
# ============================================================

table1 = pd.DataFrame([
    {
        "Modality": "Fingerprint",
        "Plain EER": fp_ckks["plain_eer"],
        "True CKKS EER": fp_ckks["ckks_eer"],
        "Plain AUC": fp_ckks["plain_auc"],
        "True CKKS AUC": fp_ckks["ckks_auc"],
        "Plain Accuracy": fp_ckks["plain_accuracy"],
        "True CKKS Accuracy": fp_ckks["ckks_accuracy"],
        "Mean Abs. Diff.": fp_ckks["mean_abs_diff_vs_plain"],
    },
    {
        "Modality": "Face",
        "Plain EER": face_ckks["plain_eer"],
        "True CKKS EER": face_ckks["ckks_eer"],
        "Plain AUC": face_ckks["plain_auc"],
        "True CKKS AUC": face_ckks["ckks_auc"],
        "Plain Accuracy": face_ckks["plain_accuracy"],
        "True CKKS Accuracy": face_ckks["ckks_accuracy"],
        "Mean Abs. Diff.": face_ckks["mean_abs_diff_vs_plain"],
    },
    {
        "Modality": "Iris",
        "Plain EER": iris_ckks["plain_eer"],
        "True CKKS EER": iris_ckks["ckks_eer"],
        "Plain AUC": iris_ckks["plain_auc"],
        "True CKKS AUC": iris_ckks["ckks_auc"],
        "Plain Accuracy": iris_ckks["plain_accuracy"],
        "True CKKS Accuracy": iris_ckks["ckks_accuracy"],
        "Mean Abs. Diff.": iris_ckks["mean_abs_diff_vs_plain"],
    },
])

# ============================================================
# 5. TABLE 2 - RECOMMENDED CONFIG UNIMODAL TRUE CKKS
# ============================================================

table2 = pd.DataFrame([
    {
        "Modality": "Fingerprint",
        "Recommended Dim": fp_rec["recommended_dim"],
        "Binary": fp_rec["recommended_binary"],
        "Plain EER": fp_rec["plain_eer"],
        "True CKKS EER": fp_rec["ckks_eer"],
        "Plain AUC": fp_rec["plain_auc"],
        "True CKKS AUC": fp_rec["ckks_auc"],
        "Plain Accuracy": fp_rec["plain_accuracy"],
        "True CKKS Accuracy": fp_rec["ckks_accuracy"],
        "Mean Abs. Diff.": fp_rec["mean_abs_diff_vs_plain"],
    },
    {
        "Modality": "Face",
        "Recommended Dim": face_rec["recommended_dim"],
        "Binary": face_rec["recommended_binary"],
        "Plain EER": face_rec["plain_eer"],
        "True CKKS EER": face_rec["ckks_eer"],
        "Plain AUC": face_rec["plain_auc"],
        "True CKKS AUC": face_rec["ckks_auc"],
        "Plain Accuracy": face_rec["plain_accuracy"],
        "True CKKS Accuracy": face_rec["ckks_accuracy"],
        "Mean Abs. Diff.": face_rec["mean_abs_diff_vs_plain"],
    },
    {
        "Modality": "Iris",
        "Recommended Dim": iris_rec["recommended_dim"],
        "Binary": iris_rec["recommended_binary"],
        "Plain EER": iris_rec["plain_eer"],
        "True CKKS EER": iris_rec["ckks_eer"],
        "Plain AUC": iris_rec["plain_auc"],
        "True CKKS AUC": iris_rec["ckks_auc"],
        "Plain Accuracy": iris_rec["plain_accuracy"],
        "True CKKS Accuracy": iris_rec["ckks_accuracy"],
        "Mean Abs. Diff.": iris_rec["mean_abs_diff_vs_plain"],
    },
])

# pretty binary text
table2["Binary"] = table2["Binary"].map(lambda x: "Binary" if x else "Non-binary")

# ============================================================
# 6. TABLE 3 - RECOMMENDED CONFIG FUSION PLAIN VS TRUE CKKS
# ============================================================

plain_fusion = rec_fusion[rec_fusion["mode"] == "recommended_plain"].copy()
ckks_fusion  = rec_fusion[rec_fusion["mode"] == "recommended_true_ckks"].copy()

plain_fusion = plain_fusion.rename(columns={
    "eer": "Plain EER",
    "auc": "Plain AUC",
    "accuracy": "Plain Accuracy"
})
ckks_fusion = ckks_fusion.rename(columns={
    "eer": "True CKKS EER",
    "auc": "True CKKS AUC",
    "accuracy": "True CKKS Accuracy"
})

table3 = plain_fusion[["fusion_name", "weights", "Plain EER", "Plain AUC", "Plain Accuracy"]].merge(
    ckks_fusion[["fusion_name", "True CKKS EER", "True CKKS AUC", "True CKKS Accuracy"]],
    on="fusion_name"
)

table3["Fusion"] = table3["fusion_name"].map({
    "face_fingerprint": "Face + Fingerprint",
    "face_iris": "Face + Iris",
    "fingerprint_iris": "Fingerprint + Iris",
    "all_three": "All Three"
})

table3["Mean |Δ EER|"] = np.abs(table3["Plain EER"] - table3["True CKKS EER"])
table3["Mean |Δ AUC|"] = np.abs(table3["Plain AUC"] - table3["True CKKS AUC"])
table3["Mean |Δ Accuracy|"] = np.abs(table3["Plain Accuracy"] - table3["True CKKS Accuracy"])

table3 = table3[[
    "Fusion", "weights",
    "Plain EER", "True CKKS EER", "Mean |Δ EER|",
    "Plain AUC", "True CKKS AUC", "Mean |Δ AUC|",
    "Plain Accuracy", "True CKKS Accuracy", "Mean |Δ Accuracy|"
]].rename(columns={"weights": "Weights"})

# ============================================================
# 7. TABLE 4 - BINARY VS NON-BINARY TRADEOFF
# ============================================================

table4_rows = []

for mod in ["Fingerprint", "Face", "Iris"]:
    sub = bin_all[bin_all["Modality"] == mod].copy()

    nb = sub[sub["Template Type"] == "Non-binary"].iloc[0]
    b  = sub[sub["Template Type"] == "Binary"].iloc[0]

    table4_rows.append({
        "Modality": mod,
        "Recommended Dim": int(nb["Dim"]),

        "Non-binary Same-key EER": float(nb["Same-key EER"]),
        "Binary Same-key EER": float(b["Same-key EER"]),

        "Non-binary Cross-key AUC": float(nb["Cross-key AUC"]),
        "Binary Cross-key AUC": float(b["Cross-key AUC"]),

        "Non-binary Pearson Corr": float(nb["Pearson Corr"]),
        "Binary Pearson Corr": float(b["Pearson Corr"]),

        "Non-binary Mean NN Overlap": float(nb["Mean NN Overlap"]),
        "Binary Mean NN Overlap": float(b["Mean NN Overlap"]),

        "Non-binary Recon Cosine": float(nb["Mean Reconstruction Cosine"]),
        "Binary Recon Cosine": float(b["Mean Reconstruction Cosine"]),
    })

table4 = pd.DataFrame(table4_rows)

# ============================================================
# 8. DISPLAY TABLES
# ============================================================

print("\n" + "="*120)
print("TABLE 1. UNIMODAL PLAIN VS TRUE CKKS")
print("="*120)
display(round_df(table1, sci_cols=["Mean Abs. Diff."]))

print("\n" + "="*120)
print("TABLE 2. RECOMMENDED-CONFIG UNIMODAL PLAIN VS TRUE CKKS")
print("="*120)
display(round_df(table2, sci_cols=["Mean Abs. Diff."]))

print("\n" + "="*120)
print("TABLE 3. RECOMMENDED-CONFIG FUSION PLAIN VS TRUE CKKS")
print("="*120)
display(round_df(table3, sci_cols=["Mean |Δ EER|", "Mean |Δ AUC|", "Mean |Δ Accuracy|"]))

print("\n" + "="*120)
print("TABLE 4. BINARY VS NON-BINARY UTILITY-IRREVERSIBILITY TRADEOFF")
print("="*120)
display(round_df(table4))

# ============================================================
# 9. SAVE TABLES
# ============================================================

table1.to_csv(os.path.join(SAVE_ROOT, "table1_unimodal_plain_vs_trueckks.csv"), index=False)
table2.to_csv(os.path.join(SAVE_ROOT, "table2_recommended_unimodal_plain_vs_trueckks.csv"), index=False)
table3.to_csv(os.path.join(SAVE_ROOT, "table3_recommended_fusion_plain_vs_trueckks.csv"), index=False)
table4.to_csv(os.path.join(SAVE_ROOT, "table4_binary_vs_nonbinary_tradeoff.csv"), index=False)

round_df(table1, sci_cols=["Mean Abs. Diff."]).to_csv(os.path.join(SAVE_ROOT, "table1_display.csv"), index=False)
round_df(table2, sci_cols=["Mean Abs. Diff."]).to_csv(os.path.join(SAVE_ROOT, "table2_display.csv"), index=False)
round_df(table3, sci_cols=["Mean |Δ EER|", "Mean |Δ AUC|", "Mean |Δ Accuracy|"]).to_csv(os.path.join(SAVE_ROOT, "table3_display.csv"), index=False)
round_df(table4).to_csv(os.path.join(SAVE_ROOT, "table4_display.csv"), index=False)

# ============================================================
# 10. CLEAN PAPER GRAPHS
# ============================================================

# -------- Graph 1: Recommended-config fusion EER --------
plot_fusion = rec_fusion[rec_fusion["mode"] == "recommended_true_ckks"].copy()
plot_fusion["Fusion"] = plot_fusion["fusion_name"].map({
    "face_fingerprint": "Face + Fingerprint",
    "face_iris": "Face + Iris",
    "fingerprint_iris": "Fingerprint + Iris",
    "all_three": "All Three"
})

plt.figure(figsize=(8.5, 5))
plt.bar(plot_fusion["Fusion"], plot_fusion["eer"])
plt.ylabel("EER")
plt.title("Recommended-Config Fusion Performance")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig1_recommended_fusion_eer.png"), dpi=300, bbox_inches="tight")
plt.show()

# -------- Graph 2: Recommended-config fusion Accuracy --------
plt.figure(figsize=(8.5, 5))
plt.bar(plot_fusion["Fusion"], plot_fusion["accuracy"])
plt.ylabel("Accuracy")
plt.title("Recommended-Config Fusion Accuracy")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig2_recommended_fusion_accuracy.png"), dpi=300, bbox_inches="tight")
plt.show()

# -------- Graph 3: Binary vs Non-binary same-key EER --------
mods = table4["Modality"].tolist()
x = np.arange(len(mods))
w = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - w/2, table4["Non-binary Same-key EER"], width=w, label="Non-binary")
plt.bar(x + w/2, table4["Binary Same-key EER"], width=w, label="Binary")
plt.xticks(x, mods)
plt.ylabel("Same-key EER")
plt.title("Utility Tradeoff: Non-binary vs Binary")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig3_binary_vs_nonbinary_samekey_eer.png"), dpi=300, bbox_inches="tight")
plt.show()

# -------- Graph 4: Binary vs Non-binary reconstruction cosine --------
plt.figure(figsize=(8, 5))
plt.bar(x - w/2, table4["Non-binary Recon Cosine"], width=w, label="Non-binary")
plt.bar(x + w/2, table4["Binary Recon Cosine"], width=w, label="Binary")
plt.xticks(x, mods)
plt.ylabel("Mean Reconstruction Cosine")
plt.title("Irreversibility Comparison: Non-binary vs Binary")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig4_binary_vs_nonbinary_recon_cosine.png"), dpi=300, bbox_inches="tight")
plt.show()

# -------- Graph 5: Binary vs Non-binary NN overlap --------
plt.figure(figsize=(8, 5))
plt.bar(x - w/2, table4["Non-binary Mean NN Overlap"], width=w, label="Non-binary")
plt.bar(x + w/2, table4["Binary Mean NN Overlap"], width=w, label="Binary")
plt.xticks(x, mods)
plt.ylabel("Mean Top-k NN Overlap")
plt.title("Neighborhood Preservation: Non-binary vs Binary")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig5_binary_vs_nonbinary_nn_overlap.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 11. TEXT SUMMARY
# ============================================================

summary_text = """
FINAL MASTER RESULTS PACKAGE

Table 1: Unimodal plain vs true CKKS
Table 2: Recommended-config unimodal plain vs true CKKS
Table 3: Recommended-config fusion plain vs true CKKS
Table 4: Binary vs non-binary utility-irreversibility tradeoff

Figures:
fig1_recommended_fusion_eer.png
fig2_recommended_fusion_accuracy.png
fig3_binary_vs_nonbinary_samekey_eer.png
fig4_binary_vs_nonbinary_recon_cosine.png
fig5_binary_vs_nonbinary_nn_overlap.png
"""

with open(os.path.join(SAVE_ROOT, "master_results_summary.txt"), "w") as f:
    f.write(summary_text)

print("\nAll final master results saved in:", SAVE_ROOT)

# ================= NOTEBOOK CELL 117 =================
# ============================================================
# RECOMMENDED-CONFIG FUSION WEIGHT RE-OPTIMIZATION
# Validation-based weight search -> Test evaluation
# ============================================================

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

SAVE_ROOT = "/kaggle/working/reoptimized_recommended_fusion"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ============================================================
# 1. PATHS
# ============================================================

# validation scores (just generated correctly)
FP_VAL   = "/kaggle/working/recommended_validation_scores/fingerprint_recommended_val_scores.csv"
FACE_VAL = "/kaggle/working/recommended_validation_scores/face_recommended_val_scores.csv"
IRIS_VAL = "/kaggle/working/recommended_validation_scores/iris_recommended_val_scores.csv"

# test scores (already generated earlier for recommended configs, plain domain)
FP_TEST   = "/kaggle/working/fingerprint_recommended_outputs/fingerprint_recommended_plain_scores.csv"
FACE_TEST = "/kaggle/working/face_recommended_outputs/face_recommended_plain_scores.csv"
IRIS_TEST = "/kaggle/working/iris_recommended_outputs/iris_recommended_plain_scores.csv"

# ============================================================
# 2. HELPERS
# ============================================================

def load_score_file(path, modality_name):
    df = pd.read_csv(path)
    keep_cols = ["pair_id", "subject1", "subject2", "idx1", "idx2", "label", "score"]
    df = df[keep_cols].copy()
    return df.rename(columns={"score": f"score_{modality_name}"})

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

    return {
        "eer": eer,
        "threshold": thr,
        "auc": auc_val,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

def generate_weight_grid(n, step=0.05):
    vals = np.arange(0, 1 + 1e-9, step)
    combos = []
    for tup in itertools.product(vals, repeat=n):
        if abs(sum(tup) - 1.0) < 1e-8:
            combos.append(tuple(round(float(x), 4) for x in tup))
    return combos

# ============================================================
# 3. LOAD + MERGE VALIDATION SCORES
# ============================================================

merge_keys = ["pair_id", "subject1", "subject2", "idx1", "idx2", "label"]

val_face = load_score_file(FACE_VAL, "face")
val_fp   = load_score_file(FP_VAL, "fingerprint")
val_iris = load_score_file(IRIS_VAL, "iris")

val_all = val_face.merge(val_fp, on=merge_keys).merge(val_iris, on=merge_keys)

for col in ["score_face", "score_fingerprint", "score_iris"]:
    val_all[f"{col}_norm"] = normalize_scores(val_all[col])

print("Validation merged preview:")
print(val_all.head())

# ============================================================
# 4. LOAD + MERGE TEST SCORES
# ============================================================

test_face = load_score_file(FACE_TEST, "face")
test_fp   = load_score_file(FP_TEST, "fingerprint")
test_iris = load_score_file(IRIS_TEST, "iris")

test_all = test_face.merge(test_fp, on=merge_keys).merge(test_iris, on=merge_keys)

for col in ["score_face", "score_fingerprint", "score_iris"]:
    test_all[f"{col}_norm"] = normalize_scores(test_all[col])

print("\nTest merged preview:")
print(test_all.head())

# ============================================================
# 5. WEIGHT SEARCH
# ============================================================

fusion_sets = {
    "face_fingerprint": ["face", "fingerprint"],
    "face_iris": ["face", "iris"],
    "fingerprint_iris": ["fingerprint", "iris"],
    "all_three": ["face", "fingerprint", "iris"]
}

search_rows = []
final_rows = []

for fusion_name, mods in fusion_sets.items():
    print(f"\nSearching weights for: {fusion_name}")

    y_val = val_all["label"].astype(int).values
    val_mat = np.column_stack([val_all[f"score_{m}_norm"].values for m in mods])

    grid = generate_weight_grid(len(mods), step=0.05)

    best_weights = None
    best_metrics = None

    for weights in grid:
        fused_val = np.sum(val_mat * np.array(weights).reshape(1, -1), axis=1)
        metrics = compute_metrics(y_val, fused_val)

        search_rows.append({
            "fusion_name": fusion_name,
            "modalities": ",".join(mods),
            "weights": ",".join([f"{m}:{w:.2f}" for m, w in zip(mods, weights)]),
            "val_eer": metrics["eer"],
            "val_auc": metrics["auc"],
            "val_accuracy": metrics["accuracy"]
        })

        if best_metrics is None or metrics["eer"] < best_metrics["eer"]:
            best_metrics = metrics
            best_weights = weights

    print("Best weights:", best_weights)
    print("Best validation EER:", best_metrics["eer"])

    # evaluate on test
    y_test = test_all["label"].astype(int).values
    test_mat = np.column_stack([test_all[f"score_{m}_norm"].values for m in mods])
    fused_test = np.sum(test_mat * np.array(best_weights).reshape(1, -1), axis=1)

    test_metrics = compute_metrics(y_test, fused_test)

    final_rows.append({
        "fusion_name": fusion_name,
        "modalities": ",".join(mods),
        "best_weights": ",".join([f"{m}:{w:.2f}" for m, w in zip(mods, best_weights)]),
        "val_eer": best_metrics["eer"],
        "test_eer": test_metrics["eer"],
        "test_auc": test_metrics["auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1_score": test_metrics["f1_score"]
    })

    pd.DataFrame({
        "pair_id": test_all["pair_id"],
        "label": test_all["label"],
        "fused_score": fused_test
    }).to_csv(os.path.join(SAVE_ROOT, f"{fusion_name}_reoptimized_test_scores.csv"), index=False)

# ============================================================
# 6. SAVE TABLES
# ============================================================

search_df = pd.DataFrame(search_rows)
search_df.to_csv(os.path.join(SAVE_ROOT, "all_weight_search_results.csv"), index=False)

final_df = pd.DataFrame(final_rows)
final_df.to_csv(os.path.join(SAVE_ROOT, "best_reoptimized_weight_results.csv"), index=False)

print("\n===== BEST RE-OPTIMIZED WEIGHTS RESULTS =====")
print(final_df)

# ============================================================
# 7. GRAPHS
# ============================================================

plot_df = final_df.copy()
plot_df["Fusion"] = plot_df["fusion_name"].map({
    "face_fingerprint": "Face + Fingerprint",
    "face_iris": "Face + Iris",
    "fingerprint_iris": "Fingerprint + Iris",
    "all_three": "All Three"
})

plt.figure(figsize=(8.5, 5))
plt.bar(plot_df["Fusion"], plot_df["test_eer"])
plt.xticks(rotation=20, ha="right")
plt.ylabel("Test EER")
plt.title("Re-Optimized Recommended-Config Fusion")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig_reoptimized_fusion_eer.png"), dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8.5, 5))
plt.bar(plot_df["Fusion"], plot_df["test_accuracy"])
plt.xticks(rotation=20, ha="right")
plt.ylabel("Test Accuracy")
plt.title("Re-Optimized Fusion Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "fig_reoptimized_fusion_accuracy.png"), dpi=300, bbox_inches="tight")
plt.show()

print("\nAll outputs saved in:", SAVE_ROOT)

# ================= NOTEBOOK CELL 122 =================
import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================
# FINAL SELECTED PATHS
# =========================================================
ROOT = Path("/kaggle/input/datasets/radhe11/backup/kaggle/working")
OUT = Path("/kaggle/working/final_paper_pipeline")
OUT.mkdir(parents=True, exist_ok=True)

FACE_EMB = ROOT / "face_pretrained_embeddings" / "face_all_embeddings_pretrained.npy"
FACE_META = ROOT / "face_pretrained_embeddings" / "face_all_embeddings_meta_pretrained.csv"

FINGER_EMB = ROOT / "fingerprint_balanced_test_outputs" / "fingerprint_all_embeddings.npy"
FINGER_META = ROOT / "fingerprint_balanced_test_outputs" / "fingerprint_all_embeddings_meta.csv"

IRIS_EMB = ROOT / "iris_balanced_test_outputs" / "iris_all_embeddings.npy"
IRIS_META = ROOT / "iris_balanced_test_outputs" / "iris_all_embeddings_meta.csv"

VAL_PAIRS = ROOT / "common_pairs_balanced" / "val_pairs_common_balanced.csv"
TEST_PAIRS = ROOT / "common_pairs_balanced" / "test_pairs_common_balanced.csv"

# =========================================================
# HELPERS
# =========================================================
def load_emb_meta(emb_path, meta_path, name):
    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path)

    print("\n" + "="*90)
    print(name.upper())
    print("="*90)
    print("Embedding path:", emb_path)
    print("Meta path     :", meta_path)
    print("Embedding shape:", emb.shape)
    print("Embedding dtype:", emb.dtype)
    print("Meta shape     :", meta.shape)
    print("Meta columns   :", list(meta.columns))

    return emb, meta

def audit_embedding_array(name, emb):
    print(f"\n{name} ARRAY AUDIT")
    print("-" * 50)
    print("NaN count:", np.isnan(emb).sum())
    print("Inf count:", np.isinf(emb).sum())
    print("Min value:", np.min(emb))
    print("Max value:", np.max(emb))

    if emb.ndim == 2:
        norms = np.linalg.norm(emb, axis=1)
        print("Norm min :", norms.min())
        print("Norm max :", norms.max())
        print("Norm mean:", norms.mean())
        print("Norm std :", norms.std())
    else:
        print("WARNING: embedding array is not 2D")

def audit_meta_match(name, emb, meta):
    print(f"\n{name} META MATCH AUDIT")
    print("-" * 50)
    print("Embedding rows:", len(emb))
    print("Meta rows     :", len(meta))
    print("Rows match?   :", len(emb) == len(meta))

    required_cols = {"subject", "img_idx", "image_path"}
    print("Has required columns?:", required_cols.issubset(set(meta.columns)))

    if required_cols.issubset(set(meta.columns)):
        print("\nSample meta rows:")
        print(meta.head(5))

        # duplicate check
        dup_count = meta.duplicated(subset=["subject", "img_idx"]).sum()
        print("\nDuplicate (subject, img_idx) rows:", dup_count)

        # unique counts
        print("Unique subjects:", meta["subject"].nunique())
        print("Unique img_idx  :", meta["img_idx"].nunique())

def audit_pairs_file(path, name):
    df = pd.read_csv(path)
    print("\n" + "="*90)
    print(name.upper())
    print("="*90)
    print("Path   :", path)
    print("Shape  :", df.shape)
    print("Columns:", list(df.columns))
    print("\nHead:")
    print(df.head())

    print("\nLabel counts:")
    print(df["label"].value_counts(dropna=False))

    print("\nUnique subject1:", df["subject1"].nunique())
    print("Unique subject2:", df["subject2"].nunique())

    return df

def check_pairs_against_meta(pairs, meta, name):
    """
    Verify whether (subject, img_idx) combinations used in pairs exist in meta.
    Pair semantics:
      left item  -> (subject1, idx1)
      right item -> (subject2, idx2)
    """
    print("\n" + "="*90)
    print(f"{name.upper()} PAIR-META CONSISTENCY")
    print("="*90)

    meta_keys = set(zip(meta["subject"].astype(int), meta["img_idx"].astype(int)))

    left_keys = list(zip(pairs["subject1"].astype(int), pairs["idx1"].astype(int)))
    right_keys = list(zip(pairs["subject2"].astype(int), pairs["idx2"].astype(int)))

    left_missing = sum(k not in meta_keys for k in left_keys)
    right_missing = sum(k not in meta_keys for k in right_keys)

    print("Left side missing keys :", left_missing)
    print("Right side missing keys:", right_missing)

    if left_missing == 0 and right_missing == 0:
        print("PASS: All pair keys exist in meta.")
    else:
        print("WARNING: Some pair keys do not exist in meta.")

def save_summary(face_emb, finger_emb, iris_emb, face_meta, finger_meta, iris_meta, val_pairs, test_pairs):
    rows = []
    for name, emb, meta in [
        ("face", face_emb, face_meta),
        ("fingerprint", finger_emb, finger_meta),
        ("iris", iris_emb, iris_meta),
    ]:
        norms = np.linalg.norm(emb, axis=1) if emb.ndim == 2 else np.array([np.nan])
        rows.append({
            "modality": name,
            "emb_rows": len(emb),
            "emb_dim": emb.shape[1] if emb.ndim == 2 else None,
            "meta_rows": len(meta),
            "rows_match": len(emb) == len(meta),
            "nan_count": int(np.isnan(emb).sum()),
            "inf_count": int(np.isinf(emb).sum()),
            "norm_min": float(np.nanmin(norms)),
            "norm_max": float(np.nanmax(norms)),
            "norm_mean": float(np.nanmean(norms)),
            "unique_subjects": int(meta["subject"].nunique()),
            "unique_img_idx": int(meta["img_idx"].nunique())
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT / "embedding_audit_summary.csv", index=False)

    pair_summary = pd.DataFrame([
        {
            "split": "val",
            "rows": len(val_pairs),
            "genuine": int((val_pairs["label"] == 1).sum()),
            "impostor": int((val_pairs["label"] == 0).sum()),
            "unique_subject1": int(val_pairs["subject1"].nunique()),
            "unique_subject2": int(val_pairs["subject2"].nunique()),
        },
        {
            "split": "test",
            "rows": len(test_pairs),
            "genuine": int((test_pairs["label"] == 1).sum()),
            "impostor": int((test_pairs["label"] == 0).sum()),
            "unique_subject1": int(test_pairs["subject1"].nunique()),
            "unique_subject2": int(test_pairs["subject2"].nunique()),
        }
    ])
    pair_summary.to_csv(OUT / "pair_audit_summary.csv", index=False)

    print("\nSaved:")
    print(OUT / "embedding_audit_summary.csv")
    print(OUT / "pair_audit_summary.csv")

# =========================================================
# LOAD
# =========================================================
face_emb, face_meta = load_emb_meta(FACE_EMB, FACE_META, "face")
finger_emb, finger_meta = load_emb_meta(FINGER_EMB, FINGER_META, "fingerprint")
iris_emb, iris_meta = load_emb_meta(IRIS_EMB, IRIS_META, "iris")

val_pairs = audit_pairs_file(VAL_PAIRS, "val pairs")
test_pairs = audit_pairs_file(TEST_PAIRS, "test pairs")

# =========================================================
# AUDIT
# =========================================================
audit_embedding_array("FACE", face_emb)
audit_embedding_array("FINGERPRINT", finger_emb)
audit_embedding_array("IRIS", iris_emb)

audit_meta_match("FACE", face_emb, face_meta)
audit_meta_match("FINGERPRINT", finger_emb, finger_meta)
audit_meta_match("IRIS", iris_emb, iris_meta)

check_pairs_against_meta(val_pairs, face_meta, "face val")
check_pairs_against_meta(test_pairs, face_meta, "face test")

check_pairs_against_meta(val_pairs, finger_meta, "fingerprint val")
check_pairs_against_meta(test_pairs, finger_meta, "fingerprint test")

check_pairs_against_meta(val_pairs, iris_meta, "iris val")
check_pairs_against_meta(test_pairs, iris_meta, "iris test")

save_summary(
    face_emb, finger_emb, iris_emb,
    face_meta, finger_meta, iris_meta,
    val_pairs, test_pairs
)

print("\nDONE: Audit completed.")

# ================= NOTEBOOK CELL 123 =================
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================================================
# PATHS
# =========================================================
ROOT = Path("/kaggle/input/datasets/radhe11/backup/kaggle/working")
OUT = Path("/kaggle/working/final_paper_pipeline/plain_unimodal_ieee")
OUT.mkdir(parents=True, exist_ok=True)

FACE_EMB = ROOT / "face_pretrained_embeddings" / "face_all_embeddings_pretrained.npy"
FACE_META = ROOT / "face_pretrained_embeddings" / "face_all_embeddings_meta_pretrained.csv"

FINGER_EMB = ROOT / "fingerprint_balanced_test_outputs" / "fingerprint_all_embeddings.npy"
FINGER_META = ROOT / "fingerprint_balanced_test_outputs" / "fingerprint_all_embeddings_meta.csv"

IRIS_EMB = ROOT / "iris_balanced_test_outputs" / "iris_all_embeddings.npy"
IRIS_META = ROOT / "iris_balanced_test_outputs" / "iris_all_embeddings_meta.csv"

VAL_PAIRS = ROOT / "common_pairs_balanced" / "val_pairs_common_balanced.csv"
TEST_PAIRS = ROOT / "common_pairs_balanced" / "test_pairs_common_balanced.csv"

# =========================================================
# HELPERS
# =========================================================
def l2_normalize(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)

def build_lookup(meta_df):
    return {(int(r.subject), int(r.img_idx)): i for i, r in meta_df.iterrows()}

def pair_scores_from_meta(emb, meta_df, pairs_df):
    emb = l2_normalize(emb.astype(np.float32))
    lookup = build_lookup(meta_df)

    idx_left = []
    idx_right = []

    for _, row in pairs_df.iterrows():
        k1 = (int(row["subject1"]), int(row["idx1"]))
        k2 = (int(row["subject2"]), int(row["idx2"]))
        idx_left.append(lookup[k1])
        idx_right.append(lookup[k2])

    a = emb[np.array(idx_left)]
    b = emb[np.array(idx_right)]
    scores = np.sum(a * b, axis=1)
    return scores

def compute_eer_metrics(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thr = float(thresholds[idx])
    return eer, thr, fpr, tpr, fnr

def tar_at_far(y_true, y_score, far_target):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = np.where(fpr <= far_target)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))

def evaluate_scores(y_true, y_score):
    eer, thr, fpr, tpr, fnr = compute_eer_metrics(y_true, y_score)
    y_pred = (y_score >= thr).astype(int)

    metrics = {
        "eer": eer,
        "threshold_at_eer": thr,
        "auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tar_far_1pct": float(tar_at_far(y_true, y_score, 0.01)),
        "tar_far_0_1pct": float(tar_at_far(y_true, y_score, 0.001)),
    }

    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "fnr": fnr
    })

    return metrics, roc_df

# =========================================================
# IEEE-STYLE PLOTTING
# =========================================================
def plot_roc(y_true, y_score, modality_name, split_name, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6.5, 5.2))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random Classifier")
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (TPR)", fontsize=11)
    plt.title(f"ROC Curve for {modality_name} Verification ({split_name})", fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_det(y_true, y_score, modality_name, split_name, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr

    plt.figure(figsize=(6.5, 5.2))
    plt.plot(fpr, fnr, linewidth=2, label="DET Curve")
    plt.xlabel("False Acceptance Rate (FAR)", fontsize=11)
    plt.ylabel("False Rejection Rate (FRR)", fontsize=11)
    plt.title(f"DET Curve for {modality_name} Verification ({split_name})", fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_score_distribution(y_true, y_score, modality_name, split_name, save_path):
    genuine = y_score[y_true == 1]
    impostor = y_score[y_true == 0]

    eer, thr, _, _, _ = compute_eer_metrics(y_true, y_score)

    plt.figure(figsize=(6.8, 5.2))
    plt.hist(impostor, bins=50, alpha=0.6, density=True, label="Impostor Scores")
    plt.hist(genuine, bins=50, alpha=0.6, density=True, label="Genuine Scores")
    plt.axvline(thr, linestyle="--", linewidth=2, label=f"EER Threshold = {thr:.4f}")
    plt.xlabel("Similarity Score", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.title(f"Score Distribution for {modality_name} Verification ({split_name})", fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# =========================================================
# LOAD DATA
# =========================================================
face_emb = np.load(FACE_EMB)
finger_emb = np.load(FINGER_EMB)
iris_emb = np.load(IRIS_EMB)

face_meta = pd.read_csv(FACE_META)
finger_meta = pd.read_csv(FINGER_META)
iris_meta = pd.read_csv(IRIS_META)

val_pairs = pd.read_csv(VAL_PAIRS)
test_pairs = pd.read_csv(TEST_PAIRS)

# =========================================================
# RUN FULL PLAIN UNIMODAL PIPELINE
# =========================================================
all_metrics = []

modalities = [
    ("Face", face_emb, face_meta, "face"),
    ("Fingerprint", finger_emb, finger_meta, "fingerprint"),
    ("Iris", iris_emb, iris_meta, "iris"),
]

for modality_title, emb, meta, modality_key in modalities:
    print("\n" + "=" * 100)
    print(f"PROCESSING {modality_title.upper()}")
    print("=" * 100)

    for split_name, pairs_df, split_key in [
        ("Validation", val_pairs, "val"),
        ("Test", test_pairs, "test")
    ]:
        print(f"\nRunning {modality_title} on {split_name} set...")

        y_true = pairs_df["label"].values.astype(int)
        scores = pair_scores_from_meta(emb, meta, pairs_df)

        # save score file
        score_df = pairs_df.copy()
        score_df["score"] = scores
        score_csv_path = OUT / f"{modality_key}_{split_key}_scores.csv"
        score_df.to_csv(score_csv_path, index=False)

        # evaluate
        metrics, roc_df = evaluate_scores(y_true, scores)
        roc_csv_path = OUT / f"{modality_key}_{split_key}_roc_points.csv"
        roc_df.to_csv(roc_csv_path, index=False)

        # save metrics
        all_metrics.append({
            "modality": modality_title,
            "split": split_name,
            **metrics
        })

        # plots
        roc_path = OUT / f"{modality_key}_{split_key}_roc.png"
        det_path = OUT / f"{modality_key}_{split_key}_det.png"
        hist_path = OUT / f"{modality_key}_{split_key}_score_distribution.png"

        plot_roc(y_true, scores, modality_title, split_name, roc_path)
        plot_det(y_true, scores, modality_title, split_name, det_path)
        plot_score_distribution(y_true, scores, modality_title, split_name, hist_path)

        # console summary
        print(f"{modality_title} | {split_name}")
        print(f"EER           : {metrics['eer']:.6f}")
        print(f"Threshold@EER : {metrics['threshold_at_eer']:.6f}")
        print(f"ROC-AUC       : {metrics['auc']:.6f}")
        print(f"Accuracy      : {metrics['accuracy']:.6f}")
        print(f"Precision     : {metrics['precision']:.6f}")
        print(f"Recall        : {metrics['recall']:.6f}")
        print(f"F1-score      : {metrics['f1']:.6f}")
        print(f"TAR@FAR=1%    : {metrics['tar_far_1pct']:.6f}")
        print(f"TAR@FAR=0.1%  : {metrics['tar_far_0_1pct']:.6f}")
        print(f"Saved score file: {score_csv_path}")
        print(f"Saved ROC file  : {roc_csv_path}")
        print(f"Saved plots     : {roc_path.name}, {det_path.name}, {hist_path.name}")

# =========================================================
# SAVE FINAL METRICS TABLE
# =========================================================
metrics_df = pd.DataFrame(all_metrics)
metrics_csv = OUT / "plain_unimodal_metrics.csv"
metrics_df.to_csv(metrics_csv, index=False)

print("\n" + "=" * 100)
print("FINAL METRICS TABLE")
print("=" * 100)
print(metrics_df)

print("\nSaved final metrics to:")
print(metrics_csv)

# Optional prettier display in Kaggle
try:
    from IPython.display import display
    display(metrics_df)
except Exception:
    pass

# ================= NOTEBOOK CELL 124 =================
import os
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================================================
# PATHS
# =========================================================
ROOT = Path("/kaggle/working/final_paper_pipeline/plain_unimodal_ieee")
OUT = Path("/kaggle/working/final_paper_pipeline/plain_fusion_ieee")
OUT.mkdir(parents=True, exist_ok=True)

# unimodal score files from previous step
FACE_VAL = ROOT / "face_val_scores.csv"
FACE_TEST = ROOT / "face_test_scores.csv"

FINGER_VAL = ROOT / "fingerprint_val_scores.csv"
FINGER_TEST = ROOT / "fingerprint_test_scores.csv"

IRIS_VAL = ROOT / "iris_val_scores.csv"
IRIS_TEST = ROOT / "iris_test_scores.csv"

# =========================================================
# HELPERS
# =========================================================
def compute_eer_metrics(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thr = float(thresholds[idx])
    return eer, thr, fpr, tpr, fnr

def tar_at_far(y_true, y_score, far_target):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = np.where(fpr <= far_target)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))

def evaluate_scores(y_true, y_score):
    eer, thr, fpr, tpr, fnr = compute_eer_metrics(y_true, y_score)
    y_pred = (y_score >= thr).astype(int)

    metrics = {
        "eer": eer,
        "threshold_at_eer": thr,
        "auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tar_far_1pct": float(tar_at_far(y_true, y_score, 0.01)),
        "tar_far_0_1pct": float(tar_at_far(y_true, y_score, 0.001)),
    }

    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "fnr": fnr
    })
    return metrics, roc_df

# =========================================================
# NORMALIZATION METHODS
# =========================================================
def fit_minmax(x):
    return {"min": float(np.min(x)), "max": float(np.max(x))}

def apply_minmax(x, params):
    mn, mx = params["min"], params["max"]
    denom = max(mx - mn, 1e-12)
    return (x - mn) / denom

def fit_zscore(x):
    return {"mean": float(np.mean(x)), "std": float(np.std(x))}

def apply_zscore(x, params):
    mean, std = params["mean"], max(params["std"], 1e-12)
    return (x - mean) / std

def fit_tanh(x):
    mean = float(np.mean(x))
    std = float(np.std(x))
    return {"mean": mean, "std": std}

def apply_tanh(x, params):
    mean, std = params["mean"], max(params["std"], 1e-12)
    z = (x - mean) / std
    return 0.5 * (np.tanh(0.01 * z) + 1.0)

def fit_normalizer(x, method):
    if method == "minmax":
        return fit_minmax(x)
    elif method == "zscore":
        return fit_zscore(x)
    elif method == "tanh":
        return fit_tanh(x)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def apply_normalizer(x, method, params):
    if method == "minmax":
        return apply_minmax(x, params)
    elif method == "zscore":
        return apply_zscore(x, params)
    elif method == "tanh":
        return apply_tanh(x, params)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

# =========================================================
# WEIGHT GRID
# =========================================================
def generate_weight_grid(n, step=0.05):
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    combos = []
    for tup in itertools.product(vals, repeat=n):
        if abs(sum(tup) - 1.0) < 1e-8:
            combos.append(tuple(float(v) for v in tup))
    return combos

# =========================================================
# PLOTTING
# =========================================================
def plot_roc(y_true, y_score, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6.5, 5.2))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random Classifier")
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (TPR)", fontsize=11)
    plt.title(title, fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_det(y_true, y_score, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr

    plt.figure(figsize=(6.5, 5.2))
    plt.plot(fpr, fnr, linewidth=2, label="DET Curve")
    plt.xlabel("False Acceptance Rate (FAR)", fontsize=11)
    plt.ylabel("False Rejection Rate (FRR)", fontsize=11)
    plt.title(title, fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_score_distribution(y_true, y_score, title, save_path):
    genuine = y_score[y_true == 1]
    impostor = y_score[y_true == 0]
    eer, thr, _, _, _ = compute_eer_metrics(y_true, y_score)

    plt.figure(figsize=(6.8, 5.2))
    plt.hist(impostor, bins=50, alpha=0.6, density=True, label="Impostor Scores")
    plt.hist(genuine, bins=50, alpha=0.6, density=True, label="Genuine Scores")
    plt.axvline(thr, linestyle="--", linewidth=2, label=f"EER Threshold = {thr:.4f}")
    plt.xlabel("Fused Similarity Score", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.title(title, fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# =========================================================
# LOAD UNIMODAL SCORES
# =========================================================
face_val = pd.read_csv(FACE_VAL)
face_test = pd.read_csv(FACE_TEST)

finger_val = pd.read_csv(FINGER_VAL)
finger_test = pd.read_csv(FINGER_TEST)

iris_val = pd.read_csv(IRIS_VAL)
iris_test = pd.read_csv(IRIS_TEST)

# sanity labels
assert np.array_equal(face_val["label"].values, finger_val["label"].values)
assert np.array_equal(face_val["label"].values, iris_val["label"].values)
assert np.array_equal(face_test["label"].values, finger_test["label"].values)
assert np.array_equal(face_test["label"].values, iris_test["label"].values)

y_val = face_val["label"].values.astype(int)
y_test = face_test["label"].values.astype(int)

val_scores_dict = {
    "face": face_val["score"].values.astype(np.float32),
    "fingerprint": finger_val["score"].values.astype(np.float32),
    "iris": iris_val["score"].values.astype(np.float32),
}

test_scores_dict = {
    "face": face_test["score"].values.astype(np.float32),
    "fingerprint": finger_test["score"].values.astype(np.float32),
    "iris": iris_test["score"].values.astype(np.float32),
}

# =========================================================
# FUSION CONFIGS
# =========================================================
fusion_configs = {
    "face+fingerprint": ["face", "fingerprint"],
    "face+iris": ["face", "iris"],
    "fingerprint+iris": ["fingerprint", "iris"],
    "face+fingerprint+iris": ["face", "fingerprint", "iris"],
}

normalization_methods = ["minmax", "zscore", "tanh"]

# =========================================================
# SEARCH BEST FUSION CONFIG ON VALIDATION
# =========================================================
search_rows = []
best_configs = []

for fusion_name, mods in fusion_configs.items():
    print("\n" + "=" * 110)
    print(f"SEARCHING BEST CONFIG FOR: {fusion_name.upper()}")
    print("=" * 110)

    n = len(mods)
    weights_grid = generate_weight_grid(n, step=0.05)

    best = None
    best_val_metrics = None
    best_val_scores = None
    best_test_scores = None
    best_params = None

    for norm_method in normalization_methods:
        # fit normalization on validation only
        fitted_params = {}
        val_norm_scores = {}
        test_norm_scores = {}

        for m in mods:
            params = fit_normalizer(val_scores_dict[m], norm_method)
            fitted_params[m] = params
            val_norm_scores[m] = apply_normalizer(val_scores_dict[m], norm_method, params)
            test_norm_scores[m] = apply_normalizer(test_scores_dict[m], norm_method, params)

        for weights in weights_grid:
            fused_val = np.zeros_like(y_val, dtype=np.float32)
            fused_test = np.zeros_like(y_test, dtype=np.float32)

            for w, m in zip(weights, mods):
                fused_val += w * val_norm_scores[m]
                fused_test += w * test_norm_scores[m]

            val_metrics, _ = evaluate_scores(y_val, fused_val)
            test_metrics, _ = evaluate_scores(y_test, fused_test)

            row = {
                "fusion_system": fusion_name,
                "normalization": norm_method,
                "weights": json.dumps(dict(zip(mods, weights))),
                "val_eer": val_metrics["eer"],
                "val_auc": val_metrics["auc"],
                "val_accuracy": val_metrics["accuracy"],
                "val_tar_far_1pct": val_metrics["tar_far_1pct"],
                "test_eer": test_metrics["eer"],
                "test_auc": test_metrics["auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_tar_far_1pct": test_metrics["tar_far_1pct"],
            }
            search_rows.append(row)

            if best is None or val_metrics["eer"] < best_val_metrics["eer"]:
                best = row
                best_val_metrics = val_metrics
                best_val_scores = fused_val.copy()
                best_test_scores = fused_test.copy()
                best_params = {
                    "fusion_name": fusion_name,
                    "mods": mods,
                    "norm_method": norm_method,
                    "weights": weights,
                    "fitted_params": fitted_params,
                }

    # save best score files
    val_out = face_val.copy()
    val_out["fused_score"] = best_val_scores
    val_score_path = OUT / f"{fusion_name.replace('+', '_')}_val_scores.csv"
    val_out.to_csv(val_score_path, index=False)

    test_out = face_test.copy()
    test_out["fused_score"] = best_test_scores
    test_score_path = OUT / f"{fusion_name.replace('+', '_')}_test_scores.csv"
    test_out.to_csv(test_score_path, index=False)

    # save roc points
    val_best_metrics, val_roc = evaluate_scores(y_val, best_val_scores)
    test_best_metrics, test_roc = evaluate_scores(y_test, best_test_scores)

    val_roc_path = OUT / f"{fusion_name.replace('+', '_')}_val_roc_points.csv"
    test_roc_path = OUT / f"{fusion_name.replace('+', '_')}_test_roc_points.csv"
    val_roc.to_csv(val_roc_path, index=False)
    test_roc.to_csv(test_roc_path, index=False)

    # plots: validation
    plot_roc(
        y_val, best_val_scores,
        f"ROC Curve for {fusion_name} Fusion (Validation)",
        OUT / f"{fusion_name.replace('+', '_')}_val_roc.png"
    )
    plot_det(
        y_val, best_val_scores,
        f"DET Curve for {fusion_name} Fusion (Validation)",
        OUT / f"{fusion_name.replace('+', '_')}_val_det.png"
    )
    plot_score_distribution(
        y_val, best_val_scores,
        f"Score Distribution for {fusion_name} Fusion (Validation)",
        OUT / f"{fusion_name.replace('+', '_')}_val_score_distribution.png"
    )

    # plots: test
    plot_roc(
        y_test, best_test_scores,
        f"ROC Curve for {fusion_name} Fusion (Test)",
        OUT / f"{fusion_name.replace('+', '_')}_test_roc.png"
    )
    plot_det(
        y_test, best_test_scores,
        f"DET Curve for {fusion_name} Fusion (Test)",
        OUT / f"{fusion_name.replace('+', '_')}_test_det.png"
    )
    plot_score_distribution(
        y_test, best_test_scores,
        f"Score Distribution for {fusion_name} Fusion (Test)",
        OUT / f"{fusion_name.replace('+', '_')}_test_score_distribution.png"
    )

    best_configs.append({
        "fusion_system": fusion_name,
        "best_normalization": best_params["norm_method"],
        "best_weights": json.dumps(dict(zip(best_params["mods"], best_params["weights"]))),

        "val_eer": val_best_metrics["eer"],
        "val_threshold_at_eer": val_best_metrics["threshold_at_eer"],
        "val_auc": val_best_metrics["auc"],
        "val_accuracy": val_best_metrics["accuracy"],
        "val_precision": val_best_metrics["precision"],
        "val_recall": val_best_metrics["recall"],
        "val_f1": val_best_metrics["f1"],
        "val_tar_far_1pct": val_best_metrics["tar_far_1pct"],
        "val_tar_far_0_1pct": val_best_metrics["tar_far_0_1pct"],

        "test_eer": test_best_metrics["eer"],
        "test_threshold_at_eer": test_best_metrics["threshold_at_eer"],
        "test_auc": test_best_metrics["auc"],
        "test_accuracy": test_best_metrics["accuracy"],
        "test_precision": test_best_metrics["precision"],
        "test_recall": test_best_metrics["recall"],
        "test_f1": test_best_metrics["f1"],
        "test_tar_far_1pct": test_best_metrics["tar_far_1pct"],
        "test_tar_far_0_1pct": test_best_metrics["tar_far_0_1pct"],
    })

    print(f"\nBEST CONFIG FOR {fusion_name}")
    print("Normalization :", best_params["norm_method"])
    print("Weights       :", dict(zip(best_params["mods"], best_params["weights"])))
    print("Validation EER:", f"{val_best_metrics['eer']:.6f}")
    print("Test EER      :", f"{test_best_metrics['eer']:.6f}")
    print("Test AUC      :", f"{test_best_metrics['auc']:.6f}")
    print("Test Accuracy :", f"{test_best_metrics['accuracy']:.6f}")
    print("Saved val score file :", val_score_path)
    print("Saved test score file:", test_score_path)

# =========================================================
# SAVE TABLES
# =========================================================
search_df = pd.DataFrame(search_rows)
search_csv = OUT / "fusion_search_all_results.csv"
search_df.to_csv(search_csv, index=False)

best_df = pd.DataFrame(best_configs)
best_csv = OUT / "fusion_best_results.csv"
best_df.to_csv(best_csv, index=False)

print("\n" + "=" * 110)
print("BEST FUSION RESULTS TABLE")
print("=" * 110)
print(best_df)

print("\nSaved:")
print(search_csv)
print(best_csv)

try:
    from IPython.display import display
    display(best_df)
except Exception:
    pass

# ================= NOTEBOOK CELL 126 =================
# =========================================================
# GPU-AWARE CKKS TEST SCRIPT
# Embeddings on GPU, CKKS on CPU (TenSEAL)
# =========================================================

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# GPU / Torch
# ----------------------------
import torch

print("=" * 90)
print("GPU CHECK")
print("=" * 90)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    DEVICE = torch.device("cuda")
else:
    print("GPU not available. Falling back to CPU.")
    DEVICE = torch.device("cpu")

# ----------------------------
# TenSEAL
# ----------------------------
try:
    import tenseal as ts
except ImportError:
    raise ImportError(
        "TenSEAL not found. Run this first in Kaggle:\n"
        "!pip install tenseal"
    )

# =========================================================
# SETTINGS
# =========================================================
DEBUG_MODE = False
DEBUG_NUM_PAIRS = 100   # first run on small sample
RUN_MODALITIES = ["face", "fingerprint", "iris"]   # can keep one only if needed

# =========================================================
# PATHS
# =========================================================
ROOT = Path("/kaggle/input/datasets/radhe11/backup/kaggle/working")
OUT = Path("/kaggle/working/final_paper_pipeline/ckks_gpu_test")
OUT.mkdir(parents=True, exist_ok=True)

PATHS = {
    "face": {
        "emb": ROOT / "face_pretrained_embeddings" / "face_all_embeddings_pretrained.npy",
        "meta": ROOT / "face_pretrained_embeddings" / "face_all_embeddings_meta_pretrained.csv",
    },
    "fingerprint": {
        "emb": ROOT / "fingerprint_balanced_test_outputs" / "fingerprint_all_embeddings.npy",
        "meta": ROOT / "fingerprint_balanced_test_outputs" / "fingerprint_all_embeddings_meta.csv",
    },
    "iris": {
        "emb": ROOT / "iris_balanced_test_outputs" / "iris_all_embeddings.npy",
        "meta": ROOT / "iris_balanced_test_outputs" / "iris_all_embeddings_meta.csv",
    }
}

VAL_PAIRS = ROOT / "common_pairs_balanced" / "val_pairs_common_balanced.csv"
TEST_PAIRS = ROOT / "common_pairs_balanced" / "test_pairs_common_balanced.csv"

# =========================================================
# HELPERS
# =========================================================
def l2_normalize_np(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)

def build_lookup(meta_df):
    return {(int(r.subject), int(r.img_idx)): i for i, r in meta_df.iterrows()}

def prepare_pair_indices(meta_df, pairs_df):
    lookup = build_lookup(meta_df)

    left_idx = []
    right_idx = []

    for _, row in pairs_df.iterrows():
        k1 = (int(row["subject1"]), int(row["idx1"]))
        k2 = (int(row["subject2"]), int(row["idx2"]))
        left_idx.append(lookup[k1])
        right_idx.append(lookup[k2])

    return np.array(left_idx, dtype=np.int64), np.array(right_idx, dtype=np.int64)

def compute_eer_metrics(y_true, y_score):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thr = float(thresholds[idx])
    return eer, thr, fpr, tpr, fnr

def tar_at_far(y_true, y_score, far_target):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = np.where(fpr <= far_target)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))

def evaluate_scores(y_true, y_score):
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

    eer, thr, fpr, tpr, fnr = compute_eer_metrics(y_true, y_score)
    y_pred = (y_score >= thr).astype(int)

    return {
        "eer": eer,
        "threshold_at_eer": thr,
        "auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tar_far_1pct": float(tar_at_far(y_true, y_score, 0.01)),
        "tar_far_0_1pct": float(tar_at_far(y_true, y_score, 0.001)),
    }

# =========================================================
# CKKS
# =========================================================
def create_ckks_context():
    private_context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    private_context.global_scale = 2 ** 40
    private_context.generate_galois_keys()

    secret_key = private_context.secret_key()

    public_context = private_context.copy()
    public_context.make_context_public()

    return private_context, public_context, secret_key

def ckks_dot_score(vec1_np, vec2_np, public_context, secret_key):
    """
    CKKS encrypted-domain score:
    encrypt vec1, multiply/dot with plaintext vec2, decrypt scalar
    """
    t0 = time.time()
    enc_v1 = ts.ckks_vector(public_context, vec1_np.tolist())
    t_enc = time.time() - t0

    t1 = time.time()
    enc_dot = enc_v1.dot(vec2_np.tolist())
    t_match = time.time() - t1

    t2 = time.time()
    score = enc_dot.decrypt(secret_key)[0]
    t_dec = time.time() - t2

    return float(score), t_enc, t_match, t_dec, (t_enc + t_match + t_dec)

# =========================================================
# GPU PLAIN SANITY
# =========================================================
def gpu_plain_scores(emb_torch, left_idx, right_idx):
    with torch.no_grad():
        a = emb_torch[left_idx]
        b = emb_torch[right_idx]
        scores = torch.sum(a * b, dim=1)
    return scores.detach().cpu().numpy()

# =========================================================
# CKKS SCORING LOOP
# =========================================================
def ckks_scores_from_gpu_embeddings(emb_torch, left_idx, right_idx, public_context, secret_key):
    """
    Embeddings stay on GPU, but each pair is moved to CPU numpy for TenSEAL.
    This tests GPU-aware pipeline, though CKKS remains CPU-side.
    """
    scores = []
    enc_times, match_times, dec_times, total_times = [], [], [], []

    for i1, i2 in zip(left_idx, right_idx):
        vec1 = emb_torch[i1].detach().cpu().numpy().astype(np.float32)
        vec2 = emb_torch[i2].detach().cpu().numpy().astype(np.float32)

        s, t_enc, t_match, t_dec, t_total = ckks_dot_score(vec1, vec2, public_context, secret_key)

        scores.append(s)
        enc_times.append(t_enc)
        match_times.append(t_match)
        dec_times.append(t_dec)
        total_times.append(t_total)

    runtime = {
        "encryption_time_sec": float(np.sum(enc_times)),
        "matching_time_sec": float(np.sum(match_times)),
        "decryption_time_sec": float(np.sum(dec_times)),
        "total_time_sec": float(np.sum(total_times)),
        "avg_time_per_pair_ms": float(np.mean(total_times) * 1000.0),
        "num_pairs": int(len(scores))
    }

    return np.array(scores, dtype=np.float32), runtime

# =========================================================
# LOAD PAIRS
# =========================================================
val_pairs = pd.read_csv(VAL_PAIRS)
test_pairs = pd.read_csv(TEST_PAIRS)

if DEBUG_MODE:
    val_pairs = val_pairs.head(DEBUG_NUM_PAIRS).copy()
    test_pairs = test_pairs.head(DEBUG_NUM_PAIRS).copy()

y_val = val_pairs["label"].values.astype(int)
y_test = test_pairs["label"].values.astype(int)

print("=" * 90)
print("PAIR INFO")
print("=" * 90)
print("Validation pairs:", len(val_pairs))
print("Test pairs      :", len(test_pairs))
print("Validation label counts:\n", val_pairs["label"].value_counts())
print("Test label counts:\n", test_pairs["label"].value_counts())

# =========================================================
# CREATE CKKS CONTEXT
# =========================================================
print("=" * 90)
print("CREATING CKKS CONTEXT")
print("=" * 90)
private_context, public_context, secret_key = create_ckks_context()
print("CKKS context ready.")

# =========================================================
# MAIN LOOP
# =========================================================
all_metrics = []
all_runtime = []

for modality in RUN_MODALITIES:
    print("\n" + "=" * 90)
    print(f"PROCESSING MODALITY: {modality.upper()}")
    print("=" * 90)

    emb_np = np.load(PATHS[modality]["emb"]).astype(np.float32)
    meta_df = pd.read_csv(PATHS[modality]["meta"])

    # normalize and move to GPU
    emb_np = l2_normalize_np(emb_np)
    emb_torch = torch.from_numpy(emb_np).to(DEVICE)

    print("Embeddings shape:", emb_np.shape)
    print("Embeddings device:", emb_torch.device)

    # pair index mapping
    val_left, val_right = prepare_pair_indices(meta_df, val_pairs)
    test_left, test_right = prepare_pair_indices(meta_df, test_pairs)

    val_left_t = torch.from_numpy(val_left).to(DEVICE)
    val_right_t = torch.from_numpy(val_right).to(DEVICE)
    test_left_t = torch.from_numpy(test_left).to(DEVICE)
    test_right_t = torch.from_numpy(test_right).to(DEVICE)

    # ----------------------------
    # GPU plain sanity scores
    # ----------------------------
    print("Running GPU plain sanity check...")
    val_plain_scores = gpu_plain_scores(emb_torch, val_left_t, val_right_t)
    test_plain_scores = gpu_plain_scores(emb_torch, test_left_t, test_right_t)

    val_plain_metrics = evaluate_scores(y_val, val_plain_scores)
    test_plain_metrics = evaluate_scores(y_test, test_plain_scores)

    print("GPU Plain Validation EER:", f"{val_plain_metrics['eer']:.6f}")
    print("GPU Plain Test EER      :", f"{test_plain_metrics['eer']:.6f}")

    # ----------------------------
    # CKKS scores
    # ----------------------------
    print("Running CKKS encrypted-domain matching...")
    val_ckks_scores, val_runtime = ckks_scores_from_gpu_embeddings(
        emb_torch, val_left, val_right, public_context, secret_key
    )
    test_ckks_scores, test_runtime = ckks_scores_from_gpu_embeddings(
        emb_torch, test_left, test_right, public_context, secret_key
    )

    val_ckks_metrics = evaluate_scores(y_val, val_ckks_scores)
    test_ckks_metrics = evaluate_scores(y_test, test_ckks_scores)

    print("CKKS Validation EER:", f"{val_ckks_metrics['eer']:.6f}")
    print("CKKS Test EER      :", f"{test_ckks_metrics['eer']:.6f}")
    print("CKKS Test AUC      :", f"{test_ckks_metrics['auc']:.6f}")
    print("CKKS Test Accuracy :", f"{test_ckks_metrics['accuracy']:.6f}")
    print("CKKS Runtime (test):", test_runtime)

    # save scores
    val_out = val_pairs.copy()
    val_out["plain_gpu_score"] = val_plain_scores
    val_out["ckks_score"] = val_ckks_scores
    val_out.to_csv(OUT / f"{modality}_val_gpu_ckks_scores.csv", index=False)

    test_out = test_pairs.copy()
    test_out["plain_gpu_score"] = test_plain_scores
    test_out["ckks_score"] = test_ckks_scores
    test_out.to_csv(OUT / f"{modality}_test_gpu_ckks_scores.csv", index=False)

    # metrics
    all_metrics.append({
        "modality": modality,
        "split": "val",
        "plain_gpu_eer": val_plain_metrics["eer"],
        "plain_gpu_auc": val_plain_metrics["auc"],
        "ckks_eer": val_ckks_metrics["eer"],
        "ckks_auc": val_ckks_metrics["auc"],
        "ckks_accuracy": val_ckks_metrics["accuracy"],
        "ckks_tar_far_1pct": val_ckks_metrics["tar_far_1pct"],
        "ckks_tar_far_0_1pct": val_ckks_metrics["tar_far_0_1pct"],
    })
    all_metrics.append({
        "modality": modality,
        "split": "test",
        "plain_gpu_eer": test_plain_metrics["eer"],
        "plain_gpu_auc": test_plain_metrics["auc"],
        "ckks_eer": test_ckks_metrics["eer"],
        "ckks_auc": test_ckks_metrics["auc"],
        "ckks_accuracy": test_ckks_metrics["accuracy"],
        "ckks_tar_far_1pct": test_ckks_metrics["tar_far_1pct"],
        "ckks_tar_far_0_1pct": test_ckks_metrics["tar_far_0_1pct"],
    })

    all_runtime.append({
        "system": f"{modality}_val_ckks",
        **val_runtime
    })
    all_runtime.append({
        "system": f"{modality}_test_ckks",
        **test_runtime
    })

# =========================================================
# SAVE TABLES
# =========================================================
metrics_df = pd.DataFrame(all_metrics)
runtime_df = pd.DataFrame(all_runtime)

metrics_df.to_csv(OUT / "gpu_ckks_metrics.csv", index=False)
runtime_df.to_csv(OUT / "gpu_ckks_runtime.csv", index=False)

print("\n" + "=" * 90)
print("FINAL METRICS")
print("=" * 90)
print(metrics_df)

print("\n" + "=" * 90)
print("FINAL RUNTIME")
print("=" * 90)
print(runtime_df)

print("\nSaved to:", OUT)

if DEBUG_MODE:
    print("\nDEBUG MODE is ON. These are only sanity-test results.")
    print("If it works, set DEBUG_MODE = False for full run.")

# ================= NOTEBOOK CELL 127 =================
# ============================================================
# FINAL CKKS BASELINE SCRIPT
# - Unimodal CKKS for Face / Fingerprint / Iris
# - Validation + Test evaluation
# - Pre-encrypt embeddings once per modality
# - Fusion CKKS using best plain-fusion weights
# - Runtime / latency analysis
# - ROC / DET / score distribution plots
# ============================================================

# If needed in Kaggle:
# !pip install tenseal

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
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
# 0. SETTINGS
# ============================================================
DEBUG_MODE = False          # True only for quick test
DEBUG_NUM_PAIRS = 200       # used only if DEBUG_MODE=True

# ============================================================
# 1. IMPORT TENSEAL
# ============================================================
import tenseal as ts

# ============================================================
# 2. PATHS
# ============================================================
ROOT = Path("/kaggle/input/datasets/radhe11/backup/kaggle/working")

OUT_DIR = Path("/kaggle/working/final_paper_pipeline/final_ckks_ieee")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# all embeddings + meta
MODALITY_PATHS = {
    "face": {
        "emb": ROOT / "face_pretrained_embeddings" / "face_all_embeddings_pretrained.npy",
        "meta": ROOT / "face_pretrained_embeddings" / "face_all_embeddings_meta_pretrained.csv",
    },
    "fingerprint": {
        "emb": ROOT / "fingerprint_balanced_test_outputs" / "fingerprint_all_embeddings.npy",
        "meta": ROOT / "fingerprint_balanced_test_outputs" / "fingerprint_all_embeddings_meta.csv",
    },
    "iris": {
        "emb": ROOT / "iris_balanced_test_outputs" / "iris_all_embeddings.npy",
        "meta": ROOT / "iris_balanced_test_outputs" / "iris_all_embeddings_meta.csv",
    },
}

VAL_PAIR_CSV = ROOT / "common_pairs_balanced" / "val_pairs_common_balanced.csv"
TEST_PAIR_CSV = ROOT / "common_pairs_balanced" / "test_pairs_common_balanced.csv"

# previous best fusion config
FUSION_BEST_CSV = Path("/kaggle/working/final_paper_pipeline/plain_fusion_ieee/fusion_best_results.csv")

# previous plain baselines for compare
PLAIN_UNIMODAL_METRICS_CSV = Path("/kaggle/working/final_paper_pipeline/plain_unimodal_ieee/plain_unimodal_metrics.csv")
PLAIN_FUSION_METRICS_CSV = Path("/kaggle/working/final_paper_pipeline/plain_fusion_ieee/fusion_best_results.csv")

# ============================================================
# 3. HELPERS
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

def evaluate_metrics(scores, labels):
    eer, threshold, fpr, tpr, thresholds = compute_eer(scores, labels)
    roc_auc = auc(fpr, tpr)

    preds = (scores >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    cm = confusion_matrix(labels, preds)
    TN, FP, FN, TP = cm.ravel()

    far = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    frr = FN / (FN + TP) if (FN + TP) > 0 else 0.0

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

    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds
    })

    return metrics, roc_df, cm

def fit_minmax(scores):
    return {"min": float(np.min(scores)), "max": float(np.max(scores))}

def apply_minmax(scores, params):
    mn = params["min"]
    mx = params["max"]
    denom = max(mx - mn, 1e-12)
    return (scores - mn) / denom

# ============================================================
# 4. PLOTTING
# ============================================================
def plot_roc_curve(fpr, tpr, roc_auc, title, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random Classifier")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_det_curve(fpr, tpr, title, save_path):
    fnr = 1 - tpr
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, fnr, linewidth=2, label="DET Curve")
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("False Rejection Rate (FRR)")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_score_distribution(scores, labels, threshold, title, save_path):
    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    bins = 30
    score_min = float(min(np.min(genuine_scores), np.min(impostor_scores)))
    score_max = float(max(np.max(genuine_scores), np.max(impostor_scores)))

    plt.figure(figsize=(7, 5))
    plt.hist(genuine_scores, bins=bins, range=(score_min, score_max),
             alpha=0.6, density=True, label="Genuine")
    plt.hist(impostor_scores, bins=bins, range=(score_min, score_max),
             alpha=0.6, density=True, label="Impostor")
    plt.axvline(threshold, linestyle="--", linewidth=2, label=f"Threshold@EER = {threshold:.4f}")
    plt.xlabel("Decrypted CKKS Similarity Score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
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
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_runtime_bar(runtime_df, title, save_path):
    plt.figure(figsize=(9, 5.5))
    x = np.arange(len(runtime_df))
    plt.bar(x, runtime_df["total_time_sec"].astype(float).values)
    plt.xticks(x, runtime_df["system"].values, rotation=30, ha="right")
    plt.ylabel("Total Time (s)")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ============================================================
# 5. CKKS CONTEXT
# ============================================================
def create_ckks_context():
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
    secret_key = secret_context.secret_key()

    return secret_context, public_context, secret_key

# ============================================================
# 6. LOAD PAIRS
# ============================================================
val_pairs = pd.read_csv(VAL_PAIR_CSV)
test_pairs = pd.read_csv(TEST_PAIR_CSV)

if DEBUG_MODE:
    val_pairs = val_pairs.head(DEBUG_NUM_PAIRS).copy()
    test_pairs = test_pairs.head(DEBUG_NUM_PAIRS).copy()
    print(f"DEBUG MODE ON: using first {DEBUG_NUM_PAIRS} pairs.")

print("Validation pairs:", val_pairs.shape)
print("Test pairs      :", test_pairs.shape)

# ============================================================
# 7. CKKS CONTEXT
# ============================================================
secret_context, public_context, secret_key = create_ckks_context()
print("CKKS context ready.")

# ============================================================
# 8. UNIMODAL CKKS
# ============================================================
unimodal_metric_rows = []
unimodal_runtime_rows = []

val_score_bank = {}
test_score_bank = {}

for modality, mp in MODALITY_PATHS.items():
    print("\n" + "=" * 100)
    print(f"PROCESSING UNIMODAL CKKS: {modality.upper()}")
    print("=" * 100)

    emb = np.load(mp["emb"]).astype(np.float64)
    meta = pd.read_csv(mp["meta"])

    # Build plaintext embedding map
    emb_map = {}
    for i, row in meta.iterrows():
        emb_map[(int(row["subject"]), int(row["img_idx"]))] = emb[i]

    # Pre-encrypt all embeddings once
    encrypted_emb_map = {}

    start_encrypt = time.time()
    for key, vec in emb_map.items():
        encrypted_emb_map[key] = ts.ckks_vector(public_context, vec.tolist())
    encrypt_time = time.time() - start_encrypt

    print(f"Encrypted {len(encrypted_emb_map)} {modality} embeddings in {encrypt_time:.2f} sec")

    for split_name, pair_df in [("val", val_pairs), ("test", test_pairs)]:
        scores = []
        labels = []
        valid_rows = []

        start_match = time.time()
        decrypt_time_acc = 0.0

        for _, row in pair_df.iterrows():
            pair_id = int(row["pair_id"])
            s1 = int(row["subject1"])
            s2 = int(row["subject2"])
            idx1 = int(row["idx1"])
            idx2 = int(row["idx2"])
            label = int(row["label"])

            k1 = (s1, idx1)
            k2 = (s2, idx2)

            if k1 not in encrypted_emb_map or k2 not in emb_map:
                continue

            enc_e1 = encrypted_emb_map[k1]
            e2 = emb_map[k2]

            enc_score = enc_e1.dot(e2.tolist())

            t_dec_0 = time.time()
            dec_score = enc_score.decrypt(secret_key)
            decrypt_time_acc += (time.time() - t_dec_0)

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

        score_csv = OUT_DIR / f"{modality}_{split_name}_scores_ckks.csv"
        score_df.to_csv(score_csv, index=False)

        scores = np.array(scores, dtype=np.float64)
        labels = np.array(labels, dtype=np.int32)

        if len(scores) == 0:
            raise ValueError(f"No valid CKKS scores for {modality} {split_name}")

        metrics, roc_df, cm = evaluate_metrics(scores, labels)

        roc_csv = OUT_DIR / f"{modality}_{split_name}_roc_points_ckks.csv"
        roc_df.to_csv(roc_csv, index=False)

        metrics_row = {
            "modality": modality,
            "split": split_name,
            "scheme": "CKKS",
            "library": "TenSEAL",
            "num_pairs_evaluated": int(len(score_df)),
            "embedding_dim": int(emb.shape[1]),
            **metrics
        }
        unimodal_metric_rows.append(metrics_row)

        runtime_row = {
            "system": f"{modality}_{split_name}_ckks",
            "modality": modality,
            "split": split_name,
            "encrypt_time_sec": float(encrypt_time),
            "matching_time_sec": float(match_time),
            "decryption_time_sec": float(decrypt_time_acc),
            "total_time_sec": float(encrypt_time + match_time),
            "avg_time_per_pair_ms": float((encrypt_time + match_time) / max(len(score_df), 1) * 1000.0),
            "num_pairs": int(len(score_df))
        }
        unimodal_runtime_rows.append(runtime_row)

        # score banks for fusion
        if split_name == "val":
            val_score_bank[modality] = scores
        else:
            test_score_bank[modality] = scores

        # plots
        fpr = roc_df["fpr"].values
        tpr = roc_df["tpr"].values
        roc_auc = metrics["roc_auc"]
        thr = metrics["threshold_at_eer"]

        plot_roc_curve(
            fpr, tpr, roc_auc,
            f"ROC Curve for {modality.capitalize()} Verification under CKKS ({split_name.capitalize()})",
            PLOT_DIR / f"{modality}_{split_name}_roc_ckks.png"
        )
        plot_det_curve(
            fpr, tpr,
            f"DET Curve for {modality.capitalize()} Verification under CKKS ({split_name.capitalize()})",
            PLOT_DIR / f"{modality}_{split_name}_det_ckks.png"
        )
        plot_score_distribution(
            scores, labels, thr,
            f"Score Distribution for {modality.capitalize()} Verification under CKKS ({split_name.capitalize()})",
            PLOT_DIR / f"{modality}_{split_name}_score_distribution_ckks.png"
        )
        plot_confusion_matrix(
            cm,
            f"{modality.capitalize()} CKKS Confusion Matrix ({split_name.capitalize()})",
            PLOT_DIR / f"{modality}_{split_name}_confusion_matrix_ckks.png"
        )

        print(f"{modality.upper()} | {split_name.upper()}")
        print(f"EER            : {metrics['eer']:.6f}")
        print(f"ROC-AUC        : {metrics['roc_auc']:.6f}")
        print(f"Accuracy       : {metrics['accuracy']:.6f}")
        print(f"TAR@FAR=1%     : {metrics['tar_at_far_1_percent']:.6f}")
        print(f"TAR@FAR=0.1%   : {metrics['tar_at_far_0_1_percent']:.6f}")
        print(f"Saved scores   : {score_csv}")
        print(f"Saved ROC CSV  : {roc_csv}")

unimodal_metrics_df = pd.DataFrame(unimodal_metric_rows)
unimodal_runtime_df = pd.DataFrame(unimodal_runtime_rows)

unimodal_metrics_df.to_csv(OUT_DIR / "unimodal_ckks_metrics.csv", index=False)
unimodal_runtime_df.to_csv(OUT_DIR / "unimodal_ckks_runtime.csv", index=False)

# ============================================================
# 9. FUSION CKKS
# ============================================================
fusion_best_df = pd.read_csv(FUSION_BEST_CSV)

fusion_metric_rows = []
fusion_runtime_rows = []

for _, row in fusion_best_df.iterrows():
    fusion_system = row["fusion_system"]
    best_weights = json.loads(row["best_weights"])

    mods = list(best_weights.keys())

    # fit minmax on validation scores
    fitted_norm = {}
    val_norm_scores = {}
    test_norm_scores = {}

    for m in mods:
        params = fit_minmax(val_score_bank[m])
        fitted_norm[m] = params
        val_norm_scores[m] = apply_minmax(val_score_bank[m], params)
        test_norm_scores[m] = apply_minmax(test_score_bank[m], params)

    val_fused = np.zeros_like(next(iter(val_norm_scores.values())))
    test_fused = np.zeros_like(next(iter(test_norm_scores.values())))

    for m, w in best_weights.items():
        val_fused += float(w) * val_norm_scores[m]
        test_fused += float(w) * test_norm_scores[m]

    # evaluate val
    val_labels = val_pairs["label"].values.astype(int)
    val_metrics, val_roc_df, val_cm = evaluate_metrics(val_fused, val_labels)

    # evaluate test
    test_labels = test_pairs["label"].values.astype(int)
    test_metrics, test_roc_df, test_cm = evaluate_metrics(test_fused, test_labels)

    # save scores
    val_score_df = val_pairs.copy()
    val_score_df["fused_score"] = val_fused
    val_score_df.to_csv(OUT_DIR / f"{fusion_system.replace('+','_')}_val_scores_ckks.csv", index=False)

    test_score_df = test_pairs.copy()
    test_score_df["fused_score"] = test_fused
    test_score_df.to_csv(OUT_DIR / f"{fusion_system.replace('+','_')}_test_scores_ckks.csv", index=False)

    # save roc points
    val_roc_df.to_csv(OUT_DIR / f"{fusion_system.replace('+','_')}_val_roc_points_ckks.csv", index=False)
    test_roc_df.to_csv(OUT_DIR / f"{fusion_system.replace('+','_')}_test_roc_points_ckks.csv", index=False)

    # metric rows
    fusion_metric_rows.append({
        "fusion_system": fusion_system,
        "split": "val",
        "best_weights": json.dumps(best_weights),
        **val_metrics
    })
    fusion_metric_rows.append({
        "fusion_system": fusion_system,
        "split": "test",
        "best_weights": json.dumps(best_weights),
        **test_metrics
    })

    # runtime: sum relevant unimodal runtimes
    rel_runtime = unimodal_runtime_df[
        (unimodal_runtime_df["split"] == "test") &
        (unimodal_runtime_df["modality"].isin(mods))
    ]

    fusion_runtime_rows.append({
        "system": f"{fusion_system}_test_ckks",
        "fusion_system": fusion_system,
        "encrypt_time_sec": float(rel_runtime["encrypt_time_sec"].sum()),
        "matching_time_sec": float(rel_runtime["matching_time_sec"].sum()),
        "decryption_time_sec": float(rel_runtime["decryption_time_sec"].sum()),
        "total_time_sec": float(rel_runtime["total_time_sec"].sum()),
        "avg_time_per_pair_ms": float(rel_runtime["avg_time_per_pair_ms"].sum()),
        "num_pairs": int(len(test_pairs))
    })

    # plots test only
    fpr = test_roc_df["fpr"].values
    tpr = test_roc_df["tpr"].values
    roc_auc = test_metrics["roc_auc"]
    thr = test_metrics["threshold_at_eer"]

    plot_roc_curve(
        fpr, tpr, roc_auc,
        f"ROC Curve for {fusion_system} Fusion under CKKS (Test)",
        PLOT_DIR / f"{fusion_system.replace('+','_')}_test_roc_ckks.png"
    )
    plot_det_curve(
        fpr, tpr,
        f"DET Curve for {fusion_system} Fusion under CKKS (Test)",
        PLOT_DIR / f"{fusion_system.replace('+','_')}_test_det_ckks.png"
    )
    plot_score_distribution(
        test_fused, test_labels, thr,
        f"Score Distribution for {fusion_system} Fusion under CKKS (Test)",
        PLOT_DIR / f"{fusion_system.replace('+','_')}_test_score_distribution_ckks.png"
    )
    plot_confusion_matrix(
        test_cm,
        f"{fusion_system} CKKS Confusion Matrix (Test)",
        PLOT_DIR / f"{fusion_system.replace('+','_')}_test_confusion_matrix_ckks.png"
    )

    print("\n" + "-" * 100)
    print(f"FUSION CKKS: {fusion_system}")
    print(f"Test EER        : {test_metrics['eer']:.6f}")
    print(f"Test ROC-AUC    : {test_metrics['roc_auc']:.6f}")
    print(f"Test Accuracy   : {test_metrics['accuracy']:.6f}")
    print(f"Test TAR@FAR=1% : {test_metrics['tar_at_far_1_percent']:.6f}")
    print(f"Weights         : {best_weights}")

fusion_metrics_df = pd.DataFrame(fusion_metric_rows)
fusion_runtime_df = pd.DataFrame(fusion_runtime_rows)

fusion_metrics_df.to_csv(OUT_DIR / "fusion_ckks_metrics.csv", index=False)
fusion_runtime_df.to_csv(OUT_DIR / "fusion_ckks_runtime.csv", index=False)

# ============================================================
# 10. PLAIN vs CKKS COMPARISON
# ============================================================
if not DEBUG_MODE:
    plain_uni_df = pd.read_csv(PLAIN_UNIMODAL_METRICS_CSV)
    plain_uni_test = plain_uni_df[plain_uni_df["split"].str.lower() == "test"].copy()
    plain_uni_test["modality"] = plain_uni_test["modality"].str.lower()

    ckks_uni_test = unimodal_metrics_df[unimodal_metrics_df["split"] == "test"].copy()

    uni_compare = plain_uni_test.merge(
        ckks_uni_test,
        on="modality",
        suffixes=("_plain", "_ckks")
    )
    uni_compare.to_csv(OUT_DIR / "plain_vs_ckks_unimodal_compare.csv", index=False)

    plain_fusion_df = pd.read_csv(PLAIN_FUSION_METRICS_CSV)
    plain_fusion_small = plain_fusion_df[[
        "fusion_system", "test_eer", "test_auc", "test_accuracy",
        "test_tar_far_1pct", "test_tar_far_0_1pct"
    ]].copy()

    plain_fusion_small = plain_fusion_small.rename(columns={
        "test_eer": "test_eer_plain",
        "test_auc": "test_auc_plain",
        "test_accuracy": "test_accuracy_plain",
        "test_tar_far_1pct": "test_tar_far_1pct_plain",
        "test_tar_far_0_1pct": "test_tar_far_0_1pct_plain"
    })

    ckks_fusion_test = fusion_metrics_df[fusion_metrics_df["split"] == "test"].copy()
    ckks_fusion_test = ckks_fusion_test.rename(columns={
        "eer": "test_eer_ckks",
        "roc_auc": "test_auc_ckks",
        "accuracy": "test_accuracy_ckks",
        "tar_at_far_1_percent": "test_tar_far_1pct_ckks",
        "tar_at_far_0_1_percent": "test_tar_far_0_1pct_ckks"
    })

    fusion_compare = plain_fusion_small.merge(
        ckks_fusion_test[[
            "fusion_system",
            "test_eer_ckks",
            "test_auc_ckks",
            "test_accuracy_ckks",
            "test_tar_far_1pct_ckks",
            "test_tar_far_0_1pct_ckks"
        ]],
        on="fusion_system"
    )
    fusion_compare.to_csv(OUT_DIR / "plain_vs_ckks_fusion_compare.csv", index=False)

# ============================================================
# 11. RUNTIME PLOTS
# ============================================================
plot_runtime_bar(
    unimodal_runtime_df[unimodal_runtime_df["split"] == "test"],
    "Runtime Analysis for Unimodal CKKS Verification (Test)",
    PLOT_DIR / "unimodal_ckks_runtime_bar.png"
)

plot_runtime_bar(
    fusion_runtime_df,
    "Runtime Analysis for Fusion CKKS Verification (Test)",
    PLOT_DIR / "fusion_ckks_runtime_bar.png"
)

# ============================================================
# 12. DISPLAY
# ============================================================
print("\n" + "=" * 100)
print("UNIMODAL CKKS METRICS")
print("=" * 100)
print(unimodal_metrics_df)

print("\n" + "=" * 100)
print("UNIMODAL CKKS RUNTIME")
print("=" * 100)
print(unimodal_runtime_df)

print("\n" + "=" * 100)
print("FUSION CKKS METRICS")
print("=" * 100)
print(fusion_metrics_df)

print("\n" + "=" * 100)
print("FUSION CKKS RUNTIME")
print("=" * 100)
print(fusion_runtime_df)

if not DEBUG_MODE:
    print("\n" + "=" * 100)
    print("PLAIN vs CKKS UNIMODAL COMPARE")
    print("=" * 100)
    print(uni_compare)

    print("\n" + "=" * 100)
    print("PLAIN vs CKKS FUSION COMPARE")
    print("=" * 100)
    print(fusion_compare)

print("\nSaved all outputs to:")
print(OUT_DIR)

if DEBUG_MODE:
    print("\nDEBUG MODE is ON. Do not use these metrics/plots in the paper.")
    print("Set DEBUG_MODE = False for final paper results.")
