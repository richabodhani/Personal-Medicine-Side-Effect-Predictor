import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from data_store import log_prediction

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Medicine AI", page_icon="💊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.big-title { font-family:'DM Serif Display',serif; font-size:42px; font-weight:400; line-height:1.15; color:#111827; margin-bottom:0.25rem; }
.big-title em { font-style:italic; color:#1D9E75; }
.subtitle { font-size:15px; color:#6B7280; margin-bottom:1.5rem; }
.badge { display:inline-block; background:#E1F5EE; color:#0F6E56; font-size:11px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; padding:5px 14px; border-radius:20px; margin-bottom:0.75rem; border:0.5px solid #9FE1CB; }
.sec-header { font-size:11px; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#6B7280; margin:1.5rem 0 0.75rem; border-bottom:0.5px solid #E5E7EB; padding-bottom:0.4rem; }
.effect-card { background:#F9FAFB; border:0.5px solid #E5E7EB; border-radius:14px; padding:0.9rem 1.1rem; margin-bottom:0.5rem; }
.effect-high   { border-left:3px solid #D85A30; border-radius:0 14px 14px 0; }
.effect-medium { border-left:3px solid #BA7517; border-radius:0 14px 14px 0; }
.effect-low    { border-left:3px solid #1D9E75; border-radius:0 14px 14px 0; }
.tag-high { background:#FAECE7; color:#993C1D; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.tag-med  { background:#FAEEDA; color:#854F0B; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.tag-low  { background:#E1F5EE; color:#0F6E56; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.model-badge { display:inline-block; background:#EEEDFE; color:#534AB7; font-size:11px; font-weight:600; padding:3px 12px; border-radius:20px; border:0.5px solid #CECBF6; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SETTINGS — tune these if still crashing
# ─────────────────────────────────────────────
FILE_PATH  = "medicine_dataset.csv"
SAMPLE_N   = 25000   # 25k rows for fast training & light RAM
TOP_N      = 30      # top 30 most frequent side effects
N_TREES    = 40      # balanced speed & performance for all models

FEATURE_COLS_WANTED = [
    "Chemical Class", "Therapeutic Class", "Action Class",
    "use0", "use1", "Habit Forming", "substitute0", "substitute1"
]

# ─────────────────────────────────────────────
# LOAD — only needed columns + sample rows
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="📂 Loading dataset (sampled for performance)...")
def load_data():
    if not os.path.isfile(FILE_PATH):
        return None

    # Step 1: peek at column names only
    peek = pd.read_csv(FILE_PATH, nrows=1, dtype=str)
    peek.columns = peek.columns.str.strip()
    all_cols = list(peek.columns)

    se_cols = sorted(
        [c for c in all_cols if c.lower().startswith("sideeffect")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or "0")
    )
    feat_cols = [c for c in FEATURE_COLS_WANTED if c in all_cols]
    use_cols  = feat_cols + se_cols
    if "Therapeutic Class" in all_cols and "Therapeutic Class" not in use_cols:
        use_cols.append("Therapeutic Class")

    # Step 2: load only needed columns + sample
    df = pd.read_csv(
        FILE_PATH,
        usecols=use_cols,
        dtype=str,
        na_values=["", "nan", "NaN", "NULL", "null"],
        low_memory=True
    )
    df.columns = df.columns.str.strip()

    # Step 3: sample to reduce RAM further
    if len(df) > SAMPLE_N:
        df = df.sample(n=SAMPLE_N, random_state=42).reset_index(drop=True)

    return df

# ─────────────────────────────────────────────
# TRAIN — ultra low RAM settings
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Training ML models — please wait...")
def train_all(_df):
    se_cols = sorted(
        [c for c in _df.columns if c.lower().startswith("sideeffect")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or "0")
    )

    def collect(row):
        return [str(v).strip().lower() for v in row
                if pd.notna(v) and str(v).strip() not in ("", "nan")]

    _df = _df.copy()
    _df["all_se"] = _df[se_cols].apply(collect, axis=1)

    # Count without building matrix
    all_effects  = [e for lst in _df["all_se"] for e in lst]
    freq_counter = Counter(all_effects)
    top_classes  = [e for e, _ in freq_counter.most_common(TOP_N)]
    top_freq_df  = pd.DataFrame(freq_counter.most_common(20),
                                columns=["Side Effect", "Count"])

    # Binarize only top N
    mlb = MultiLabelBinarizer(classes=top_classes)
    y   = mlb.fit_transform(_df["all_se"])

    # Convert to int8 to save RAM (binary labels need only 0/1)
    y = y.astype(np.int8)

    # Features
    FEATURE_COLS = [c for c in FEATURE_COLS_WANTED if c in _df.columns]
    X = _df[FEATURE_COLS].fillna("Unknown").astype(str)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Encode
    encoder     = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_enc = encoder.fit_transform(X_train).astype(np.float32)  # float32 = half RAM of float64
    X_test_enc  = encoder.transform(X_test).astype(np.float32)

    # Scaling for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled  = scaler.transform(X_test_enc)

    # Models — all 6 models with optimized settings
    rf = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=N_TREES, random_state=42,
                               n_jobs=-1, max_depth=15, min_samples_leaf=5)
    )
    dt = MultiOutputClassifier(
        DecisionTreeClassifier(random_state=42, max_depth=15, min_samples_leaf=5)
    )
    lr = MultiOutputClassifier(
        LogisticRegression(max_iter=300, solver="saga", C=1.0, class_weight="balanced", n_jobs=-1, random_state=42)
    )
    xgb = MultiOutputClassifier(
        XGBClassifier(n_estimators=N_TREES, max_depth=6, learning_rate=0.1, subsample=0.8,
                      colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)
    )
    lgbm = MultiOutputClassifier(
        LGBMClassifier(n_estimators=N_TREES, max_depth=8, learning_rate=0.1, num_leaves=63,
                       subsample=0.8, colsample_bytree=0.8, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
    )
    cat = MultiOutputClassifier(
        CatBoostClassifier(iterations=N_TREES, depth=6, learning_rate=0.1, random_seed=42,
                           verbose=0, thread_count=-1, auto_class_weights="Balanced")
    )

    rf.fit(X_train_enc, y_train)
    dt.fit(X_train_enc, y_train)
    lr.fit(X_train_scaled, y_train)
    xgb.fit(X_train_enc, y_train)
    lgbm.fit(X_train_enc, y_train)
    cat.fit(X_train_enc, y_train)

    # Metrics
    def get_metrics(y_true, y_pred):
        hl = hamming_loss(y_true, y_pred)
        return {
            "Accuracy":     round(1 - hl, 4),
            "F1 Score":     round(f1_score(y_true, y_pred, average="micro", zero_division=0), 4),
            "Precision":    round(precision_score(y_true, y_pred, average="micro", zero_division=0), 4),
            "Recall":       round(recall_score(y_true, y_pred, average="micro", zero_division=0), 4),
            "Hamming Loss": round(hl, 6),
        }

    lr_proba = np.array([est.predict_proba(X_test_scaled)[:, 1] for est in lr.estimators_]).T
    pred_lr = (lr_proba >= 0.35).astype(int)

    metrics = {
        "Random Forest":       get_metrics(y_test, rf.predict(X_test_enc)),
        "Decision Tree":       get_metrics(y_test, dt.predict(X_test_enc)),
        "Logistic Regression": get_metrics(y_test, pred_lr),
        "XGBoost":             get_metrics(y_test, xgb.predict(X_test_enc)),
        "LightGBM":            get_metrics(y_test, lgbm.predict(X_test_enc)),
        "CatBoost":            get_metrics(y_test, cat.predict(X_test_enc)),
    }

    importances = np.mean([e.feature_importances_ for e in rf.estimators_], axis=0)
    feat_imp    = dict(zip(FEATURE_COLS, importances))

    return {
        "rf": rf, "dt": dt, "lr": lr,
        "xgb": xgb, "lgbm": lgbm, "cat": cat,
        "encoder": encoder, "scaler": scaler, "mlb": mlb,
        "FEATURE_COLS": FEATURE_COLS,
        "metrics": metrics, "feat_imp": feat_imp,
        "top_freq_df": top_freq_df,
        "total_unique": len(freq_counter),
        "total_rows": len(_df),
    }

# ─────────────────────────────────────────────
# DRUG DATA
# ─────────────────────────────────────────────
DRUGS = {
    "Ibuprofen":     {"chem":"NSAID",            "thera":"Analgesic",        "use":"Pain relief",         "habit":"No"},
    "Paracetamol":   {"chem":"Para-aminophenol", "thera":"Analgesic",        "use":"Fever / Pain",        "habit":"No"},
    "Aspirin":       {"chem":"Salicylate",       "thera":"Analgesic",        "use":"Pain relief",         "habit":"No"},
    "Metformin":     {"chem":"Biguanide",        "thera":"Antidiabetic",     "use":"Type 2 Diabetes",     "habit":"No"},
    "Amoxicillin":   {"chem":"Beta-lactam",      "thera":"Antibiotic",       "use":"Bacterial infection", "habit":"No"},
    "Omeprazole":    {"chem":"Benzimidazole",    "thera":"PPI",              "use":"Acid reflux / GERD",  "habit":"No"},
    "Atorvastatin":  {"chem":"Statin",           "thera":"Lipid-lowering",   "use":"High cholesterol",    "habit":"No"},
    "Amlodipine":    {"chem":"Dihydropyridine",  "thera":"Antihypertensive", "use":"Blood pressure",      "habit":"No"},
    "Cetirizine":    {"chem":"Piperazine",       "thera":"Antihistamine",    "use":"Allergies",           "habit":"No"},
    "Azithromycin":  {"chem":"Macrolide",        "thera":"Antibiotic",       "use":"Bacterial infection", "habit":"No"},
    "Metoprolol":    {"chem":"Beta-blocker",     "thera":"Antihypertensive", "use":"Heart rate / BP",     "habit":"No"},
    "Diazepam":      {"chem":"Benzodiazepine",   "thera":"Anxiolytic",       "use":"Anxiety / Seizures",  "habit":"Yes"},
    "Codeine":       {"chem":"Opiate",           "thera":"Analgesic",        "use":"Pain / Cough",        "habit":"Yes"},
    "Lisinopril":    {"chem":"ACE inhibitor",    "thera":"Antihypertensive", "use":"Blood pressure",      "habit":"No"},
    "Sertraline":    {"chem":"SSRI",             "thera":"Antidepressant",   "use":"Depression",          "habit":"No"},
    "Ciprofloxacin": {"chem":"Fluoroquinolone",  "thera":"Antibiotic",       "use":"Bacterial infection", "habit":"No"},
    "Prednisolone":  {"chem":"Glucocorticoid",   "thera":"Corticosteroid",   "use":"Inflammation",        "habit":"No"},
    "Losartan":      {"chem":"ARB",              "thera":"Antihypertensive", "use":"Blood pressure",      "habit":"No"},
    "Gabapentin":    {"chem":"GABA analogue",    "thera":"Anticonvulsant",   "use":"Neuropathic pain",    "habit":"Yes"},
    "Pantoprazole":  {"chem":"Benzimidazole",    "thera":"PPI",              "use":"Acid reflux",         "habit":"No"},
}
THERA_OPTIONS = sorted(set(v["thera"] for v in DRUGS.values()))
CHEM_OPTIONS  = sorted(set(v["chem"]  for v in DRUGS.values()))
USE_OPTIONS   = sorted(set(v["use"]   for v in DRUGS.values()))

# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
data = load_data()
if data is None:
    st.error(f"❌ Dataset not found. Place `{FILE_PATH}` in the same folder as `app.py`.")
    st.stop()

models = train_all(data)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💊 Predict Side Effects", "📊 Model Performance", "🔥 Dataset Insights"])

# ══════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="badge">ML-Powered · 100% Free · No API Key Needed</div>', unsafe_allow_html=True)
    st.markdown('<p class="big-title">Medicine <em>Side Effect</em> Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Trained on your real dataset · Random Forest · Decision Tree · Logistic Regression · XGBoost · LightGBM · CatBoost</p>', unsafe_allow_html=True)

    st.sidebar.header("👤 Patient Profile")
    user_name    = st.sidebar.text_input("Name", placeholder="John Doe")
    age          = st.sidebar.number_input("Age", 0, 120, 25)
    weight       = st.sidebar.number_input("Weight (kg)", 1, 300, 70)
    st.sidebar.markdown("---")
    model_choice = st.sidebar.selectbox("🤖 ML Model", ["Random Forest", "Decision Tree", "Logistic Regression", "XGBoost", "LightGBM", "CatBoost"])
    m            = models["metrics"][model_choice]
    st.sidebar.caption(f"Accuracy: **{m['Accuracy']}** · F1: **{m['F1 Score']}**")
    st.sidebar.markdown("---")
    st.sidebar.success("✅ No API key needed!\nUsing your trained ML model.")

    st.markdown('<p class="sec-header">Quick select medicine</p>', unsafe_allow_html=True)
    selected_drug = st.selectbox("Medicine", ["— select —"] + list(DRUGS.keys()),
                                 label_visibility="collapsed")

    if selected_drug != "— select —":
        d        = DRUGS[selected_drug]
        di_thera = THERA_OPTIONS.index(d["thera"]) if d["thera"] in THERA_OPTIONS else 0
        di_chem  = CHEM_OPTIONS.index(d["chem"])   if d["chem"]  in CHEM_OPTIONS  else 0
        di_use   = USE_OPTIONS.index(d["use"])      if d["use"]   in USE_OPTIONS   else 0
        di_habit = ["No","Yes","Unknown"].index(d["habit"])
    else:
        di_thera = di_chem = di_use = di_habit = 0

    st.markdown('<p class="sec-header">Or select details manually</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        thera = st.selectbox("Therapeutic class", ["— select —"] + THERA_OPTIONS,
                             index=di_thera+1 if selected_drug != "— select —" else 0)
        chem  = st.selectbox("Chemical class",    ["— select —"] + CHEM_OPTIONS,
                             index=di_chem+1  if selected_drug != "— select —" else 0)
    with c2:
        use   = st.selectbox("Primary use",   ["— select —"] + USE_OPTIONS,
                             index=di_use+1   if selected_drug != "— select —" else 0)
        habit = st.selectbox("Habit forming", ["No","Yes","Unknown"], index=di_habit)

    st.markdown("")
    predict_btn = st.button("🔬 Predict Side Effects", type="primary", use_container_width=True)

    if predict_btn:
        thera_val = thera if thera != "— select —" else "Unknown"
        chem_val  = chem  if chem  != "— select —" else "Unknown"
        use_val   = use   if use   != "— select —" else "Unknown"

        if thera_val == "Unknown" and chem_val == "Unknown" and selected_drug == "— select —":
            st.error("Please select a medicine or at least one detail.")
            st.stop()

        fc        = models["FEATURE_COLS"]
        input_row = {
            "Chemical Class":    chem_val,  "Therapeutic Class": thera_val,
            "Action Class":      "Unknown", "use0":              use_val,
            "use1":              "Unknown", "Habit Forming":     habit,
            "substitute0":       "Unknown", "substitute1":       "Unknown",
        }
        input_df  = pd.DataFrame([[input_row.get(c, "Unknown") for c in fc]], columns=fc)
        input_enc = models["encoder"].transform(input_df.fillna("Unknown")).astype(np.float32)

        if model_choice == "Logistic Regression":
            input_scaled = models["scaler"].transform(input_enc)
            lr_proba = np.array([est.predict_proba(input_scaled)[:, 1] for est in models["lr"].estimators_]).T
            pred = (lr_proba >= 0.35).astype(int)[0]
        else:
            model_obj = {
                "Random Forest":       models["rf"],
                "Decision Tree":       models["dt"],
                "XGBoost":             models["xgb"],
                "LightGBM":            models["lgbm"],
                "CatBoost":            models["cat"],
            }[model_choice]
            pred = model_obj.predict(input_enc)[0]
        labels = [models["mlb"].classes_[i] for i, v in enumerate(pred) if v == 1]

        def tag_sev(i, total):
            if   i < total * 0.3:  return "High"
            elif i < total * 0.65: return "Medium"
            else:                  return "Low"

        sev_score = min(10, max(1, round(len(labels) / 3)))
        se_list   = [{"name": e, "severity": tag_sev(i, max(len(labels), 1)), "description": ""}
                     for i, e in enumerate(labels[:12])]

        log_prediction(
            patient_name=user_name, age=age, weight=weight,
            drug=selected_drug if selected_drug != "— select —" else thera_val,
            thera=thera_val, chem=chem_val, use=use_val, habit=habit,
            severity_score=sev_score, total_effects=len(labels),
            side_effects=se_list,
            summary=f"{model_choice} predicted {len(labels)} side effects.",
        )

        st.markdown("---")
        st.markdown("## 📋 Prediction Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Patient",        user_name or "N/A")
        m2.metric("Age / Weight",   f"{age} yrs / {weight} kg")
        m3.metric("Effects Found",  len(labels))
        m4.metric("Severity Score", f"{sev_score} / 10")

        st.markdown(f'<div class="model-badge">Model: {model_choice} · Accuracy: {models["metrics"][model_choice]["Accuracy"]} · F1: {models["metrics"][model_choice]["F1 Score"]}</div>', unsafe_allow_html=True)

        if labels:
            SEV_CSS = {"High":"effect-high","Medium":"effect-medium","Low":"effect-low"}
            TAG_CSS = {"High":"tag-high","Medium":"tag-med","Low":"tag-low"}
            st.markdown('<p class="sec-header">Predicted side effects</p>', unsafe_allow_html=True)
            for i, effect in enumerate(labels[:15]):
                sev = tag_sev(i, len(labels))
                st.markdown(f"""
                <div class="effect-card {SEV_CSS[sev]}">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <strong style="font-size:14px;text-transform:capitalize">{effect}</strong>
                        <span class="{TAG_CSS[sev]}">{sev}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<p class="sec-header">Top predicted effects chart</p>', unsafe_allow_html=True)
            chart_df = pd.DataFrame({
                "Side Effect": labels[:10],
                "Score":       np.linspace(1.0, 0.4, min(10, len(labels)))
            })
            fig, ax = plt.subplots(figsize=(8, 4))
            bar_colors = ["#D85A30" if i < 3 else "#BA7517" if i < 6 else "#1D9E75"
                          for i in range(len(chart_df))]
            ax.barh(chart_df["Side Effect"], chart_df["Score"],
                    color=bar_colors, edgecolor="white")
            ax.invert_yaxis()
            ax.set_xlabel("Relative Rank Score")
            ax.set_title(f"Top Predicted Side Effects — {model_choice}", fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No strong side effects predicted. Try selecting a different medicine or class.")

        st.warning("⚠️ For educational purposes only — not medical advice.")


# ══════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="big-title">Model <em>Performance</em></p>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">Trained on 80% · Tested on 20% · Sample of {SAMPLE_N:,} rows used</p>', unsafe_allow_html=True)

    st.markdown('<p class="sec-header">Evaluation metrics</p>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame(models["metrics"]).T.reset_index()
    metrics_df.columns = ["Model","Accuracy","F1 Score","Precision","Recall","Hamming Loss"]
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    graph_files = {
        "graph1_accuracy.png":            "Accuracy Comparison",
        "graph2_f1score.png":             "F1 Score Comparison",
        "graph3_precision.png":           "Precision Comparison",
        "graph4_recall.png":              "Recall Comparison",
        "graph5_hamming_loss.png":        "Hamming Loss",
        "graph6_all_metrics.png":         "All Metrics Side-by-Side",
        "graph7_radar.png":               "Radar Chart",
        "graph10_feature_importance.png": "Feature Importance",
    }
    existing = [(k, v) for k, v in graph_files.items() if os.path.isfile(k)]

    if existing:
        st.markdown('<p class="sec-header">Graphs from project.py</p>', unsafe_allow_html=True)
        for i in range(0, len(existing), 2):
            cols = st.columns(2)
            for j, (fname, title) in enumerate(existing[i:i+2]):
                with cols[j]:
                    st.caption(title)
                    st.image(fname, use_container_width=True)
    else:
        st.info("Run `python project.py` once to generate graphs. They will appear here automatically.")

    st.markdown('<p class="sec-header">Feature importance — Random Forest (live)</p>', unsafe_allow_html=True)
    fi = pd.DataFrame(list(models["feat_imp"].items()),
                      columns=["Feature","Importance"]).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(fi["Feature"], fi["Importance"],
            color=sns.color_palette("Blues_d", len(fi)), edgecolor="white")
    for i, (_, row) in enumerate(fi.iterrows()):
        ax.text(row["Importance"]+0.001, i, f"{row['Importance']:.4f}", va="center", fontsize=9)
    ax.set_title("Feature Importance — Random Forest", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════
# TAB 3 — DATASET INSIGHTS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="big-title">Dataset <em>Insights</em></p>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">medicine_dataset.csv · {models["total_rows"]:,} rows sampled</p>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows Sampled",        f"{models['total_rows']:,}")
    k2.metric("Unique Side Effects",  f"{models['total_unique']:,}")
    k3.metric("Features Used",        len(models["FEATURE_COLS"]))
    k4.metric("Labels Modelled",      f"Top {TOP_N}")

    st.markdown('<p class="sec-header">Top 20 most frequent side effects</p>', unsafe_allow_html=True)
    if os.path.isfile("graph8_top_side_effects.png"):
        st.image("graph8_top_side_effects.png", use_container_width=True)
    else:
        top_se = models["top_freq_df"]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top_se["Side Effect"], top_se["Count"],
                color=sns.color_palette("RdYlGn_r", len(top_se)), edgecolor="white")
        ax.invert_yaxis()
        ax.set_xlabel("Number of Medicines")
        ax.set_title("Top 20 Most Frequent Side Effects", fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    if os.path.isfile("graph9_side_effects_pie.png"):
        st.markdown('<p class="sec-header">Top 10 side effects — pie chart</p>', unsafe_allow_html=True)
        st.image("graph9_side_effects_pie.png", use_container_width=True)

    if "Therapeutic Class" in data.columns:
        st.markdown('<p class="sec-header">Therapeutic class distribution</p>', unsafe_allow_html=True)
        tc = data["Therapeutic Class"].value_counts().head(15).reset_index()
        tc.columns = ["Class","Count"]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=tc, x="Count", y="Class", palette="viridis", ax=ax)
        ax.set_title("Top 15 Therapeutic Classes in Dataset", fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.markdown("---")
st.caption("Medicine AI · Powered by your ML models · Completely Free · [Open Admin Panel](http://localhost:8502)")