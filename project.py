"""
==================================================================
  MEDICINE SIDE EFFECT PREDICTION — COMPLETE ML PROJECT
  Multi-Label Classification of Drug Side Effects Using Machine Learning
==================================================================
"""

import sys
import os
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Ensure UTF-8 output encoding for console
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss

# Gradient boosting models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Suppress warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# STYLE & PALETTES
# ─────────────────────────────────────────────────────────────
MODEL_NAMES = [
    "Random Forest",
    "Logistic Regression",
    "Decision Tree",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]
MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


# ═════════════════════════════════════════════════════════════
# 1. LOAD DATASET (.csv first, fallback to .zip)
# ═════════════════════════════════════════════════════════════
CSV_PATH = "medicine_dataset.csv"
ZIP_PATH = "archive.zip"

print("=" * 65)
print("  STEP 1: LOADING RAW DATASET")
print("=" * 65)

if os.path.exists(CSV_PATH):
    print(f"Loading from '{CSV_PATH}' ...")
    data = pd.read_csv(CSV_PATH, low_memory=False)
elif os.path.exists(ZIP_PATH):
    print(f"Extracting & Loading from '{ZIP_PATH}' ...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV found inside archive.zip")
        data = pd.read_csv(z.open(csv_files[0]), low_memory=False)
else:
    raise FileNotFoundError(f"Neither '{CSV_PATH}' nor '{ZIP_PATH}' was found.")

data.columns = data.columns.str.strip()
print(f"Dataset Loaded: {data.shape[0]:,} rows x {data.shape[1]} columns\n")


# ═════════════════════════════════════════════════════════════
# 2. FULL PREPROCESSING (All 58 Columns)
# ═════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 2: FULL PREPROCESSING (ALL 58 COLUMNS)")
print("=" * 65)

# STEP 1 — Strip whitespace and unify NaN across all string/object columns
obj_cols = data.select_dtypes(include=['object']).columns
for col in obj_cols:
    data[col] = data[col].astype(str).str.strip()
    data[col] = data[col].replace({
        'nan': np.nan, 'NaN': np.nan, 'NAN': np.nan,
        'none': np.nan, 'None': np.nan, 'NONE': np.nan,
        'null': np.nan, 'NULL': np.nan, '': np.nan
    })

# STEP 2 — Standardise text casing
if "Therapeutic Class" in data.columns:
    data["Therapeutic Class"] = data["Therapeutic Class"].astype(str).str.upper().str.strip().replace({'NAN': np.nan, 'NONE': np.nan, '': np.nan})
if "Chemical Class" in data.columns:
    data["Chemical Class"] = data["Chemical Class"].astype(str).str.title().str.strip().replace({'Nan': np.nan, 'None': np.nan, '': np.nan})
if "Action Class" in data.columns:
    data["Action Class"] = data["Action Class"].astype(str).str.title().str.strip().replace({'Nan': np.nan, 'None': np.nan, '': np.nan})
if "Habit Forming" in data.columns:
    data["Habit Forming"] = data["Habit Forming"].astype(str).str.capitalize().str.strip().replace({'Nan': np.nan, 'None': np.nan, '': np.nan})
if "name" in data.columns:
    data["name"] = data["name"].astype(str).str.lower().str.strip().replace({'nan': np.nan, 'none': np.nan, '': np.nan})

# STEP 3 — Standardise sideEffect columns (sideEffect0 through sideEffect41)
all_se_cols_raw = [c for c in data.columns if c.lower().startswith("sideeffect")]
for col in all_se_cols_raw:
    data[col] = data[col].astype(str).str.strip().str.lower().replace({'nan': np.nan, 'none': np.nan, '': np.nan})

# STEP 4 — Drop extremely sparse sideEffect columns (sideEffect25 to sideEffect41)
cols_to_drop = [f"sideEffect{i}" for i in range(25, 42) if f"sideEffect{i}" in data.columns]
data.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped {len(cols_to_drop)} sparse sideEffect columns (sideEffect25-sideEffect41)")
retained_se_cols = [f"sideEffect{i}" for i in range(25) if f"sideEffect{i}" in data.columns]
print(f"Retained {len(retained_se_cols)} sideEffect columns (sideEffect0-sideEffect24)")

# STEP 5 — Remove duplicate rows
before_dups = len(data)
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)
print(f"Duplicate rows removed: {before_dups - len(data)}")

# STEP 6 — Save full cleaned dataset
CLEANED_CSV_PATH = "medicine_dataset_full_cleaned.csv"
data.to_csv(CLEANED_CSV_PATH, index=False)
print(f"Saved Cleaned Dataset: '{CLEANED_CSV_PATH}' ({data.shape[0]:,} rows x {data.shape[1]} columns)\n")


# ═════════════════════════════════════════════════════════════
# 3. 15 KEY FEATURES (from Dataset_Table.docx)
# ═════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 3: SELECTING 15 KEY FEATURES (6 TARGET + 9 FEATURES)")
print("=" * 65)

TARGET_SE_COLS = [
    "sideEffect0", "sideEffect1", "sideEffect2",
    "sideEffect3", "sideEffect4", "sideEffect5"
]
FEATURE_COLS = [
    "name",
    "Chemical Class",
    "Therapeutic Class",
    "Action Class",
    "use0",
    "use1",
    "Habit Forming",
    "substitute0",
    "substitute1"
]

print(f"Target Columns (6):  {', '.join(TARGET_SE_COLS)}")
print(f"Feature Columns (9): {', '.join(FEATURE_COLS)}\n")


# ═════════════════════════════════════════════════════════════
# 4. MULTI-LABEL BINARIZATION (Top-100 & Top-20 Labels)
# ═════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 4: MULTI-LABEL TARGET EXTRACTION & BINARIZATION")
print("=" * 65)

# Build per-row side effect list from sideEffect0-5
data["all_side_effects"] = data[TARGET_SE_COLS].apply(
    lambda row: [
        str(v).strip().lower()
        for v in row
        if pd.notna(v) and str(v).strip() not in ("", "nan", "none")
    ], axis=1
)

# Fit full MultiLabelBinarizer to analyze distribution
mlb_full = MultiLabelBinarizer()
y_full = mlb_full.fit_transform(data["all_side_effects"])
freq_counts = y_full.sum(axis=0)
sorted_indices = np.argsort(freq_counts)[::-1]
total_unique_ses = len(mlb_full.classes_)
print(f"Total unique side effects found across dataset: {total_unique_ses}")

# Top 100 most frequent side effects (for RF, DT, XGB, LGBM, CatBoost)
top100_indices = sorted_indices[:100]
top100_classes = [mlb_full.classes_[i] for i in top100_indices]
mlb100 = MultiLabelBinarizer(classes=top100_classes)
y100 = mlb100.fit_transform(data["all_side_effects"]).astype(np.int8)
print(f"y100 shape: {y100.shape} (Top 100 labels for tree & boosting models)")

# Top 20 most frequent side effects (for Logistic Regression)
top20_indices = sorted_indices[:20]
top20_classes = [mlb_full.classes_[i] for i in top20_indices]
mlb20 = MultiLabelBinarizer(classes=top20_classes)
y20 = mlb20.fit_transform(data["all_side_effects"]).astype(np.int8)
print(f"y20 shape : {y20.shape} (Top 20 labels for Logistic Regression)\n")

# Display confirmed Top 20 table
top20_freq_df = pd.DataFrame({
    "Rank": range(1, 21),
    "Side Effect": [c.title() for c in top20_classes],
    "Count": freq_counts[top20_indices],
    "% of Medicines": (freq_counts[top20_indices] / len(data) * 100).round(2)
})
print("TOP 20 MOST FREQUENT SIDE EFFECTS:")
print(top20_freq_df.to_string(index=False))
print()


# ═════════════════════════════════════════════════════════════
# 5. FEATURE ENCODING & DATASET SPLIT
# ═════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 5: FEATURE ENCODING & TRAIN/TEST SPLIT")
print("=" * 65)

# RULE 1: Fill NaN in feature columns ONLY after selecting 9 feature columns
X = data[FEATURE_COLS].copy().fillna("Unknown").astype(str)
print(f"X shape: {X.shape}")

# Train / Test split (80/20, random_state=42)
X_train, X_test, y100_train, y100_test, y20_train, y20_test = train_test_split(
    X, y100, y20, test_size=0.2, random_state=42
)
print(f"Train rows: {X_train.shape[0]:,} | Test rows: {X_test.shape[0]:,}")

# ENCODING PATH A — OrdinalEncoder for RF, DT, XGBoost, LightGBM, CatBoost
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_enc = encoder.fit_transform(X_train).astype(float)
X_test_enc  = encoder.transform(X_test).astype(float)

# ENCODING PATH B — StandardScaler for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc)
X_test_scaled  = scaler.transform(X_test_enc)
print("Features encoded (Path A: Ordinal, Path B: Scaled) successfully.\n")


# ═════════════════════════════════════════════════════════════
# 6. MODEL INITIALIZATION & TRAINING (6 MODELS)
# ═════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 6: TRAINING ALL 6 MACHINE LEARNING MODELS")
print("=" * 65)

# --- 1. Random Forest ---
print("[1/6] Training Random Forest (100 estimators, max_depth=30, y100) ...")
rf = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=100,
        max_depth=30,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
)
rf.fit(X_train_enc, y100_train)
print("      Random Forest trained.")

# --- 2. Logistic Regression (4 Fixes Applied) ---
print("[2/6] Training Logistic Regression (solver='saga', class_weight='balanced', y20) ...")
lr = MultiOutputClassifier(
    LogisticRegression(
        solver='saga',
        C=1.0,
        max_iter=300,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
)
lr.fit(X_train_scaled, y20_train)
print("      Logistic Regression trained.")

# --- 3. Decision Tree ---
print("[3/6] Training Decision Tree (max_depth=25, min_samples_leaf=3, y100) ...")
dt = MultiOutputClassifier(
    DecisionTreeClassifier(
        max_depth=25,
        min_samples_leaf=3,
        random_state=42
    )
)
dt.fit(X_train_enc, y100_train)
print("      Decision Tree trained.")

# --- 4. XGBoost ---
print("[4/6] Training XGBoost (100 estimators, max_depth=6, y100) ...")
xgb = MultiOutputClassifier(
    XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
)
xgb.fit(X_train_enc, y100_train)
print("      XGBoost trained.")

# --- 5. LightGBM ---
print("[5/6] Training LightGBM (100 estimators, num_leaves=63, y100) ...")
lgbm = MultiOutputClassifier(
    LGBMClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
)
lgbm.fit(X_train_enc, y100_train)
print("      LightGBM trained.")

# --- 6. CatBoost ---
print("[6/6] Training CatBoost (100 iterations, depth=6, y100) ...")
cat = MultiOutputClassifier(
    CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=0,
        thread_count=-1,
        auto_class_weights='Balanced'
    )
)
cat.fit(X_train_enc, y100_train)
print("      CatBoost trained.")
print("\nAll 6 ML models trained successfully!\n")


# ═════════════════════════════════════════════════════════════
# 7. PREDICTION & EVALUATION
# ═════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 7: MODEL EVALUATION ON TEST DATASET")
print("=" * 65)

# 1. RF Prediction
pred_rf = rf.predict(X_test_enc)

# 2. LR Prediction (FIX 4: predict_proba + threshold=0.35)
lr_proba = np.array([
    est.predict_proba(X_test_scaled)[:, 1]
    for est in lr.estimators_
]).T
pred_lr = (lr_proba >= 0.35).astype(int)

# 3. DT Prediction
pred_dt = dt.predict(X_test_enc)

# 4. XGBoost Prediction
pred_xgb = xgb.predict(X_test_enc)

# 5. LightGBM Prediction
pred_lgbm = lgbm.predict(X_test_enc)

# 6. CatBoost Prediction
pred_cat = cat.predict(X_test_enc)


def evaluate(name, y_true, y_pred):
    f1        = f1_score(y_true, y_pred, average='micro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='micro', zero_division=0)
    hl        = hamming_loss(y_true, y_pred)
    accuracy  = 1 - hl

    print(f"+{'-'*44}+")
    print(f"|  {name:<42}|")
    print(f"+{'-'*44}+")
    print(f"|  Accuracy (1-Hamming) : {accuracy:<21.4f}|")
    print(f"|  F1 Score (micro)     : {f1:<21.4f}|")
    print(f"|  Precision (micro)    : {precision:<21.4f}|")
    print(f"|  Recall (micro)       : {recall:<21.4f}|")
    print(f"|  Hamming Loss         : {hl:<21.6f}|")
    print(f"+{'-'*44}+\n")
    return f1, precision, recall, hl, accuracy


rf_m   = evaluate("Random Forest",       y100_test, pred_rf)
lr_m   = evaluate("Logistic Regression", y20_test,  pred_lr)
dt_m   = evaluate("Decision Tree",       y100_test, pred_dt)
xgb_m  = evaluate("XGBoost",             y100_test, pred_xgb)
lgbm_m = evaluate("LightGBM",            y100_test, pred_lgbm)
cat_m  = evaluate("CatBoost",            y100_test, pred_cat)

metrics_list = [rf_m, lr_m, dt_m, xgb_m, lgbm_m, cat_m]

results_df = pd.DataFrame({
    "Model": MODEL_NAMES,
    "Accuracy":     [m[4] for m in metrics_list],
    "F1 Score":     [m[0] for m in metrics_list],
    "Precision":    [m[1] for m in metrics_list],
    "Recall":       [m[2] for m in metrics_list],
    "Hamming Loss": [m[3] for m in metrics_list],
})

print("SUMMARY EVALUATION TABLE (ALL 6 MODELS):")
print(results_df.to_string(index=False))
print()


# ═════════════════════════════════════════════════════════════
# 8. FEATURE IMPORTANCES
# ═════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 8: FEATURE IMPORTANCES (RANDOM FOREST)")
print("=" * 65)

importances_rf = np.mean([est.feature_importances_ for est in rf.estimators_], axis=0)
feat_imp_df = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Importance": importances_rf
}).sort_values("Importance", ascending=False)

print("Random Forest Feature Importances:")
print(feat_imp_df.to_string(index=False))
print()


# ═════════════════════════════════════════════════════════════
# 9. GENERATING ALL 10 RESULT GRAPHS
# ═════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 9: GENERATING 10 COMPREHENSIVE RESULT GRAPHS")
print("=" * 65)

# GRAPH 1 — graph1_accuracy.png
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(results_df["Model"], results_df["Accuracy"], color=MODEL_COLORS, width=0.55, edgecolor="white")
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.012, f"{height:.4f}",
            ha="center", va="bottom", fontweight="bold", fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_title("Model Accuracy Comparison (1 - Hamming Loss)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Model")
plt.tight_layout()
plt.savefig("graph1_accuracy.png", dpi=150)
plt.close()
print("   [1/10] graph1_accuracy.png saved")

# GRAPH 2 — graph2_f1score.png
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(results_df["Model"], results_df["F1 Score"], color=MODEL_COLORS, width=0.55, edgecolor="white")
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.012, f"{height:.4f}",
            ha="center", va="bottom", fontweight="bold", fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_title("F1 Score Comparison (Micro)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("F1 Score")
ax.set_xlabel("Model")
plt.tight_layout()
plt.savefig("graph2_f1score.png", dpi=150)
plt.close()
print("   [2/10] graph2_f1score.png saved")

# GRAPH 3 — graph3_precision.png
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(results_df["Model"], results_df["Precision"], color=MODEL_COLORS, width=0.55, edgecolor="white")
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.012, f"{height:.4f}",
            ha="center", va="bottom", fontweight="bold", fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_title("Precision Comparison (Micro)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Precision")
ax.set_xlabel("Model")
plt.tight_layout()
plt.savefig("graph3_precision.png", dpi=150)
plt.close()
print("   [3/10] graph3_precision.png saved")

# GRAPH 4 — graph4_recall.png
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(results_df["Model"], results_df["Recall"], color=MODEL_COLORS, width=0.55, edgecolor="white")
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.012, f"{height:.4f}",
            ha="center", va="bottom", fontweight="bold", fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_title("Recall Comparison (Micro)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Recall")
ax.set_xlabel("Model")
plt.tight_layout()
plt.savefig("graph4_recall.png", dpi=150)
plt.close()
print("   [4/10] graph4_recall.png saved")

# GRAPH 5 — graph5_hamming.png
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(results_df["Model"], results_df["Hamming Loss"], color=MODEL_COLORS, width=0.55, edgecolor="white")
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + (max(results_df["Hamming Loss"]) * 0.02),
            f"{height:.6f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
ax.set_title("Hamming Loss Comparison (Lower = Better)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Hamming Loss")
ax.set_xlabel("Model")
plt.tight_layout()
plt.savefig("graph5_hamming.png", dpi=150)
plt.savefig("graph5_hamming_loss.png", dpi=150)
plt.close()
print("   [5/10] graph5_hamming.png saved")

# GRAPH 6 — graph6_all_metrics.png
metrics_plot = ["Accuracy", "F1 Score", "Precision", "Recall"]
metric_colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
x = np.arange(len(MODEL_NAMES))
width = 0.2

fig, ax = plt.subplots(figsize=(14, 6.5))
for i, (metric, color) in enumerate(zip(metrics_plot, metric_colors)):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, results_df[metric], width, label=metric, color=color, alpha=0.9, edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=0)

ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES, fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("All Metrics Comparison — All 6 Models", fontsize=14, fontweight="bold", pad=12)
ax.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig("graph6_all_metrics.png", dpi=150)
plt.close()
print("   [6/10] graph6_all_metrics.png saved")

# GRAPH 7 — graph7_radar.png
radar_metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]
num_vars = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(2, 3, figsize=(18, 10), subplot_kw=dict(polar=True))
axes = axes.flatten()
fig.suptitle("Model Performance Radar Comparison (All 6 Models)", fontsize=16, fontweight="bold", y=0.98)

for ax, model_name, m_vals, color in zip(axes, MODEL_NAMES, metrics_list, MODEL_COLORS):
    values = [m_vals[4], m_vals[0], m_vals[1], m_vals[2]]
    values += values[:1]
    
    ax.plot(angles, values, color=color, linewidth=2.2, label=model_name)
    ax.fill(angles, values, color=color, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title(model_name, size=12, fontweight="bold", pad=14, color=color)

plt.tight_layout()
plt.savefig("graph7_radar.png", dpi=150)
plt.close()
print("   [7/10] graph7_radar.png saved")

# GRAPH 8 — graph8_top_side_effects.png
fig, ax = plt.subplots(figsize=(12, 8))
palette = sns.color_palette("RdYlGn_r", 20)
bars = ax.barh(top20_freq_df["Side Effect"], top20_freq_df["Count"], color=palette, edgecolor="white")
ax.invert_yaxis()
for bar, pct in zip(bars, top20_freq_df["% of Medicines"]):
    ax.text(bar.get_width() + 1000, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%",
            va="center", fontsize=9, fontweight="bold", color="#333333")
ax.set_xlabel("Number of Medicines", fontsize=11)
ax.set_title(f"Top 20 Most Frequent Side Effects\n(% = share of {len(data):,} medicines)",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("graph8_top_side_effects.png", dpi=150)
plt.close()
print("   [8/10] graph8_top_side_effects.png saved")

# GRAPH 9 — graph9_pie.png
top10_freq_df = top20_freq_df.head(10)
fig, ax = plt.subplots(figsize=(9, 9))
wedges, texts, autotexts = ax.pie(
    top10_freq_df["Count"],
    labels=top10_freq_df["Side Effect"],
    autopct="%1.1f%%",
    startangle=140,
    colors=sns.color_palette("Set3", 10),
    pctdistance=0.82,
    wedgeprops=dict(edgecolor="white", linewidth=1.5)
)
for t in autotexts:
    t.set_fontsize(9)
    t.set_fontweight("bold")
for t in texts:
    t.set_fontsize(10)
ax.set_title("Top 10 Side Effects — Share Distribution", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("graph9_pie.png", dpi=150)
plt.savefig("graph9_side_effects_pie.png", dpi=150)
plt.close()
print("   [9/10] graph9_pie.png saved")

# GRAPH 10 — graph10_feature_importance.png
feat_imp_sorted = feat_imp_df.sort_values("Importance", ascending=True)
fig, ax = plt.subplots(figsize=(10, 6.5))
bars = ax.barh(feat_imp_sorted["Feature"], feat_imp_sorted["Importance"],
               color=sns.color_palette("Blues_d", len(feat_imp_sorted)), edgecolor="white")
for bar in bars:
    val = bar.get_width()
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
            va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("Mean Feature Importance", fontsize=11)
ax.set_title("Feature Importance — Random Forest (9 Selected Features)", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("graph10_feature_importance.png", dpi=150)
plt.close()
print("   [10/10] graph10_feature_importance.png saved\n")


# ═════════════════════════════════════════════════════════════
# 10. SUMMARY OF OUTPUT FILES
# ═════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 10: EXECUTION COMPLETE & OUTPUT FILES SUMMARY")
print("=" * 65)
output_files = [
    "medicine_dataset_full_cleaned.csv",
    "graph1_accuracy.png",
    "graph2_f1score.png",
    "graph3_precision.png",
    "graph4_recall.png",
    "graph5_hamming.png",
    "graph6_all_metrics.png",
    "graph7_radar.png",
    "graph8_top_side_effects.png",
    "graph9_pie.png",
    "graph10_feature_importance.png"
]
for f in output_files:
    exists = "OK" if os.path.exists(f) else "MISSING"
    size_str = f"({os.path.getsize(f)/1024:.1f} KB)" if os.path.exists(f) else ""
    print(f"  [{exists}] {f:<36} {size_str}")

print("\nPipeline finished successfully!")
