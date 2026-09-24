# 💊 Personal Medicine Side Effect Predictor
> **Multi-Label Classification of Drug Side Effects Using Machine Learning**

An end-to-end clinical machine learning application that predicts potential drug side effects from pharmacological and chemical drug properties. Includes an interactive Streamlit patient dashboard, an administrative audit panel, and a comprehensive 6-model benchmark pipeline.

---

## 🌟 Key Features

- **Multi-Label Inference**: Predicts multiple concurrent adverse effects using multi-output classification.
- **6 Machine Learning Models**:
  - 🌲 **Random Forest** (Micro-F1: 0.9689, Accuracy: 99.72%)
  - 🌳 **Decision Tree** (Micro-F1: 0.9494, Accuracy: 99.54%)
  - ⚡ **XGBoost** (Micro-F1: 0.9333, Accuracy: 99.41%)
  - 🍃 **LightGBM** (Micro-F1: 0.9230, Accuracy: 99.27%)
  - 🐱 **CatBoost** (Micro-F1: 0.7702, Accuracy: 97.48%)
  - 📈 **Logistic Regression** (Balanced weighting with threshold tuning)
- **Interactive Patient App (`app.py`)**:
  - Patient intake (Age, Weight)
  - Quick medicine selector (common drugs) or custom pharmacological property inputs
  - Side effect severity classification (High / Medium / Low)
  - Live model performance metrics table
  - Dataset distribution explorer
- **Admin & Audit Portal (`admin.py`)**:
  - Authenticated admin panel (`admin` / `medai@2024`)
  - Prediction audit logging (`data_store.py` / `predictions_log.json`)
  - Analytics KPIs and CSV export
- **Diagnostic Visualizations**: Generates 10 diagnostic graphs comparing model performance and feature importances.

---

## 📊 Benchmark Results

Evaluated across a held-out test set of **49,644 medicines** from a 248k+ drug dataset:

| Model | Accuracy (1 - Hamming) | Micro F1 Score | Micro Precision | Micro Recall | Hamming Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Random Forest** | **0.9972** | **0.9689** | **0.9780** | **0.9600** | **0.002781** |
| 🥈 **Decision Tree** | **0.9954** | **0.9494** | **0.9414** | **0.9575** | **0.004606** |
| 🥉 **XGBoost** | **0.9941** | **0.9333** | **0.9553** | **0.9122** | **0.005887** |
| 4. **LightGBM** | **0.9927** | **0.9230** | **0.8785** | **0.9722** | **0.007321** |
| 5. **CatBoost** | **0.9748** | **0.7702** | **0.6539** | **0.9367** | **0.025221** |
| 6. **Logistic Regression** | **0.5018** | **0.3876** | **0.2461** | **0.9109** | **0.498210** |

---

## ⭐ Feature Importances (Random Forest)

1. **`use0` (0.1897)** — Primary medical indication
2. **`Chemical Class` (0.1511)** — Molecular structure family
3. **`substitute0` (0.1504)** — Primary therapeutic substitute
4. **`substitute1` (0.1336)** — Secondary therapeutic substitute
5. **`Action Class` (0.1230)** — Biochemical mechanism of action
6. **`Therapeutic Class` (0.1090)** — Pharmacological category
7. **`use1` (0.0912)** — Secondary medical indication
8. **`name` (0.0379)** — Drug trade name
9. **`Habit Forming` (0.0142)** — Dependency potential

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/richabodhani/Personal-Medicine-Side-Effect-Predictor.git
cd Personal-Medicine-Side-Effect-Predictor
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run the Admin Dashboard
```bash
streamlit run admin.py --server.port 8502
```
Open [http://localhost:8502](http://localhost:8502). Default login: `admin` / `medai@2024`.

### 4. Run the Full Model Pipeline & Generate Graphs
```bash
python project.py
```

---

## 📁 Repository Structure

```text
├── app.py                            # Streamlit patient & clinician prediction UI
├── admin.py                          # Streamlit administrative dashboard & logs
├── project.py                        # Complete 6-model training and evaluation script
├── data_store.py                     # Prediction storage & JSON logger
├── predictions_log.json              # Historical prediction records
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
├── README.md                         # Documentation
└── graph*.png                        # 10 diagnostic performance and insight graphs
```

---

## ⚠️ Disclaimer
*This project is for educational and research purposes only and should not be used as professional medical advice.*
