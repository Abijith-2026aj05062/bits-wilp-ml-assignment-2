"""
BITS Pilani - WILP M.Tech (AIML)
Course: Machine Learning (Assignment - 2)
Application: app.py (Interactive Streamlit Evaluation Dashboard)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# =============================================================================
# 1. PAGE CONFIGURATION & HEADER
# =============================================================================
st.set_page_config(
    page_title="Student Performance Classifier | ML Assignment 2",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 UCI Student Performance Classification Dashboard")
st.markdown(
    "**BITS Pilani WILP — M.Tech (AIML) | Machine Learning Assignment - 2**  \n"
    "Predicting Student Academic Success (`Pass: G3 >= 10` vs. `Fail: G3 < 10`) "
    "across 5 supervised classification algorithms without interim grade leakage."
)
st.markdown("---")

# =============================================================================
# 2. ARTIFACT LOADING (PREPROCESSOR & SAVED MODELS)
# =============================================================================
MODEL_DIR = "model"
MODEL_REGISTRY = {
    "Logistic Regression": "logistic_regression_model.joblib",
    "Decision Tree": "decision_tree_model.joblib",
    "KNN": "knn_model.joblib",
    "Naive Bayes": "naive_bayes_model.joblib",
    "Support Vector Machine": "support_vector_machine_model.joblib"
}


@st.cache_resource
def load_preprocessor():
    # Load the fitted ColumnTransformer (StandardScaler + OneHotEncoder) saved during training
    prep_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    if not os.path.exists(prep_path):
        st.error(f"Missing preprocessor file at `{prep_path}`. Please run `python train_models.py` first.")
        st.stop()
    return joblib.load(prep_path)


@st.cache_resource
def load_trained_model(model_label):
    # Load the selected pre-trained classifier from disk so no retraining occurs in the app
    model_path = os.path.join(MODEL_DIR, MODEL_REGISTRY[model_label])
    if not os.path.exists(model_path):
        st.error(f"Missing model file at `{model_path}`. Please run `python train_models.py` first.")
        st.stop()
    return joblib.load(model_path)


preprocessor = load_preprocessor()

# =============================================================================
# 3. SIDEBAR: MODEL SELECTION DROPDOWN & TEST CSV UPLOADER
# =============================================================================
st.sidebar.header("🛠️ Evaluation Controls")

# Mandatory Feature (b): Model selection dropdown
selected_model_name = st.sidebar.selectbox(
    "1. Select Classification Model:",
    options=list(MODEL_REGISTRY.keys()),
    index=4  # Defaults to Support Vector Machine (Overall Winner)
)

st.sidebar.markdown("---")

# Mandatory Feature (a): Dataset upload option (CSV)
uploaded_csv = st.sidebar.file_uploader(
    "2. Upload Test Dataset (CSV):",
    type=["csv"],
    help="Upload test_data.csv (or any compatible test CSV containing input features and the 'pass' target column)."
)

if uploaded_csv is not None:
    try:
        # Handle both comma-separated test_data.csv and semicolon-separated student-por.csv
        test_df = pd.read_csv(uploaded_csv, sep=None, engine="python")
        st.sidebar.success(f"Loaded uploaded file: `{uploaded_csv.name}` ({len(test_df)} rows)")
    except Exception as exc:
        st.sidebar.error(f"Failed to read uploaded CSV: {exc}")
        st.stop()
else:
    default_csv_path = "test_data.csv"
    if os.path.exists(default_csv_path):
        test_df = pd.read_csv(default_csv_path)
        st.sidebar.info(f"Using default repository `{default_csv_path}` ({len(test_df)} rows).")
    else:
        st.error("No file uploaded and `test_data.csv` was not found in the root directory.")
        st.stop()

# If a user uploads the raw student-por.csv, derive 'pass' and drop G1, G2, G3 automatically
if "pass" not in test_df.columns and "G3" in test_df.columns:
    test_df["pass"] = (test_df["G3"] >= 10).astype(int)

for leaked_col in ["G1", "G2", "G3"]:
    if leaked_col in test_df.columns:
        test_df = test_df.drop(columns=[leaked_col])

if "pass" not in test_df.columns:
    st.error("The uploaded CSV must contain the ground-truth target column named `'pass'` (or `'G3'`).")
    st.stop()

# Separate features and target
X_test_raw = test_df.drop(columns=["pass"])
y_test = test_df["pass"].astype(int)

# Transform features using the pre-fitted training pipeline
try:
    X_test_transformed = preprocessor.transform(X_test_raw)
except Exception as exc:
    st.error(f"Feature schema mismatch during preprocessing: {exc}")
    st.stop()


# =============================================================================
# 4. HELPER FUNCTION FOR METRIC COMPUTATION
# =============================================================================
def evaluate_classifier(clf, X_matrix, y_true):
    # Generate discrete predictions and positive-class probabilities without retraining
    y_pred = clf.predict(X_matrix)
    y_prob = clf.predict_proba(X_matrix)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }
    return y_pred, y_prob, metrics


# =============================================================================
# 5. MAIN DASHBOARD TABS
# =============================================================================
tab_single, tab_benchmark, tab_preview = st.tabs([
    "🔍 Single Model Evaluation",
    "📊 All-Models Comparison Table",
    "🗂️ Test Dataset Preview"
])

# -----------------------------------------------------------------------------
# TAB 1: SINGLE MODEL EVALUATION
# -----------------------------------------------------------------------------
with tab_single:
    st.subheader(f"Evaluation Results: **{selected_model_name}**")

    active_clf = load_trained_model(selected_model_name)
    y_pred_single, y_prob_single, single_metrics = evaluate_classifier(
        active_clf, X_test_transformed, y_test
    )

    # Mandatory Feature (c): Display of all 6 evaluation metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{single_metrics['Accuracy']:.4f}")
    col2.metric("AUC Score", f"{single_metrics['AUC']:.4f}")
    col3.metric("Precision", f"{single_metrics['Precision']:.4f}")
    col4.metric("Recall", f"{single_metrics['Recall']:.4f}")
    col5.metric("F1 Score", f"{single_metrics['F1']:.4f}")
    col6.metric("MCC Score", f"{single_metrics['MCC']:.4f}")

    st.markdown("---")

    # Mandatory Feature (d): Side-by-side Confusion Matrix & Classification Report
    left_col, right_col = st.columns([1, 1.15])

    with left_col:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_single)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Predicted Fail (0)", "Predicted Pass (1)"],
            yticklabels=["Actual Fail (0)", "Actual Pass (1)"],
            annot_kws={"size": 13, "weight": "bold"},
            ax=ax
        )
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
        ax.set_title(f"{selected_model_name} — Confusion Matrix", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right_col:
        st.markdown("#### Detailed Classification Report")
        report_dict = classification_report(
            y_test,
            y_pred_single,
            target_names=["Fail (Class 0)", "Pass (Class 1)"],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(
            report_df.style.format({
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1-score": "{:.4f}",
                "support": "{:.0f}"
            }),
            use_container_width=True
        )
        st.caption(
            "**Note:** `MCC` (Matthews Correlation Coefficient) accounts for all four quadrants "
            "of the confusion matrix (`TP`, `TN`, `FP`, `FN`), providing a balanced evaluation "
            "even when Passing students outnumber Failing students."
        )

# -----------------------------------------------------------------------------
# TAB 2: MULTI-MODEL COMPARISON VIEW
# -----------------------------------------------------------------------------
with tab_benchmark:
    st.subheader("📊 Benchmark Comparison Across All 5 Classifiers")
    st.markdown(
        "The table below displays the live evaluation of all 5 saved models on the currently "
        "loaded test dataset (`N = " + str(len(y_test)) + "` instances):"
    )

    comparison_rows = []
    for model_name in MODEL_REGISTRY.keys():
        clf_item = load_trained_model(model_name)
        _, _, m_scores = evaluate_classifier(clf_item, X_test_transformed, y_test)
        comparison_rows.append({
            "ML Model Name": model_name,
            "Accuracy": round(m_scores["Accuracy"], 4),
            "AUC": round(m_scores["AUC"], 4),
            "Precision": round(m_scores["Precision"], 4),
            "Recall": round(m_scores["Recall"], 4),
            "F1": round(m_scores["F1"], 4),
            "MCC": round(m_scores["MCC"], 4)
        })

    comparison_df = pd.DataFrame(comparison_rows)

    st.dataframe(
        comparison_df.style.format({
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "MCC": "{:.4f}"
        }).highlight_max(
            subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            color="#D1FAE5"
        ),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "**Overall Winner: Support Vector Machine (RBF Kernel)** — Achieves the highest "
        "Accuracy (`0.9000`), AUC (`0.8017`), F1 Score (`0.9456`), and MCC (`0.3796`) on the test set."
    )

# -----------------------------------------------------------------------------
# TAB 3: TEST DATASET PREVIEW
# -----------------------------------------------------------------------------
with tab_preview:
    st.subheader("🗂️ Active Test Dataset Preview")
    st.write(f"**Dimensions:** `{test_df.shape[0]}` rows × `{test_df.shape[1]}` columns")
    st.dataframe(test_df.head(25), use_container_width=True)