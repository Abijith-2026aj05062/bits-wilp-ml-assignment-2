"""
BITS Pilani - WILP M.Tech (AIML)
Course: Machine Learning (Assignment - 2)
Script: train_models.py
Dataset: UCI Student Performance Dataset (student-por.csv)
Target: Binary Classification (Pass = 1 if G3 >= 10, Fail = 0 if G3 < 10)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def main():
    # =========================================================================
    # STEP 1: LOAD DATASET & DEFINE BINARY CLASSIFICATION TARGET
    # =========================================================================
    data_path = "student-por.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find '{data_path}'. Make sure 'student-por.csv' is in the current folder."
        )

    # The UCI Student Performance CSV uses semicolons (';') as column delimiters.
    df = pd.read_csv(data_path, sep=";")
    print(f"[INFO] Loaded '{data_path}' with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Thresholding Logic & Domain Formulation:
    # In the Portuguese secondary education grading system, G3 (final grade) ranges from 0 to 20,
    # where a score of 10 or higher is a passing grade.
    # Mathematical indicator function: y_i = I(G3_i >= 10) in {0, 1}
    df["pass"] = (df["G3"] >= 10).astype(int)

    # Data Leakage Prevention:
    # G1 (first period grade) and G2 (second period grade) are interim components of the final
    # grade G3 and exhibit near-collinear dependence with G3. Retaining G1 and G2 causes severe
    # target leakage, where the model merely memorizes prior grades instead of learning from
    # socio-demographic, academic support, and behavioral features. We drop G1, G2, and G3.
    leakage_columns = ["G1", "G2", "G3"]
    df_clean = df.drop(columns=leakage_columns)

    # Separate input features (X) and target vector (y)
    X = df_clean.drop(columns=["pass"])
    y = df_clean["pass"]

    # =========================================================================
    # STEP 2: IDENTIFY FEATURE TYPES & PERFORM 80/20 TRAIN-TEST SPLIT
    # =========================================================================
    # Separate nominal/categorical string columns from numeric integer/float columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"[INFO] Identified {len(numerical_cols)} numerical features and {len(categorical_cols)} categorical features.")

    # Partition dataset into 80% training (519 rows) and 20% testing (130 rows).
    # Splitting BEFORE fitting any scalers or encoders is mandatory so test set statistics
    # (sample mean, variance, or category distributions) never leak into training.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # =========================================================================
    # STEP 3: SAVE RAW TEST DATA FOR STREAMLIT UPLOAD & EVALUATION
    # =========================================================================
    # Combine raw test features with the ground-truth 'pass' column and save to root folder.
    # The Streamlit app will load this CSV, apply the saved preprocessor, and evaluate models
    # without retraining on the test split.
    test_export_df = X_test.copy()
    test_export_df["pass"] = y_test
    test_export_df.to_csv("test_data.csv", index=False)
    print(f"[INFO] Saved 'test_data.csv' ({test_export_df.shape[0]} rows, {test_export_df.shape[1]} columns).")

    # =========================================================================
    # STEP 4: FIT PREPROCESSING PIPELINE STRICTLY ON TRAINING DATA
    # =========================================================================
    # Statistical Reasoning for Preprocessing:
    # 1. StandardScaler on Numerical Features:
    #    Applies z-score standardization: z = (x - mu_train) / sigma_train
    #    Ensures zero mean and unit variance so features with larger numerical ranges (such as
    #    'absences' in [0, 93] or 'age' in [15, 22]) do not dominate Euclidean distance in KNN,
    #    RBF kernel distances in SVM, or L2-regularized weights in Logistic Regression.
    # 2. OneHotEncoder on Categorical Features:
    #    Converts nominal categories (e.g., 'Mjob', 'reason', 'guardian') into binary indicator
    #    vectors so linear and distance-based models do not assume artificial ordinal rankings.
    #    Setting handle_unknown='ignore' prevents runtime crashes if unseen categories appear at test time.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )

    # Fit strictly on X_train, then transform both X_train and X_test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Create 'model/' directory to store serialized artifacts
    os.makedirs("model", exist_ok=True)
    joblib.dump(preprocessor, os.path.join("model", "preprocessor.joblib"))
    print("[INFO] Saved fitted preprocessing pipeline to 'model/preprocessor.joblib'.")

    # =========================================================================
    # STEP 5: INSTANTIATE & TRAIN THE 5 REQUIRED CLASSIFIERS
    # =========================================================================
    models = {
        # 1. Logistic Regression:
        # Models posterior class probability via the sigmoid function:
        # P(y=1|x) = 1 / (1 + exp(-(w^T x + b)))
        # Optimizes regularized binary cross-entropy (log-loss) with L2 penalty:
        # J(w) = - (1/N) * sum[y_i log(p_i) + (1 - y_i) log(1 - p_i)] + (1 / (2C)) * ||w||_2^2
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42
        ),

        # 2. Decision Tree Classifier:
        # Recursively partitions the feature space by selecting splits that minimize Gini Impurity:
        # Gini(t) = 1 - sum_{k=0}^{1} p(k|t)^2
        # Pre-pruning with max_depth=5 prevents the tree from memorizing noisy leaf nodes (overfitting).
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            random_state=42
        ),

        # 3. K-Nearest Neighbors (KNN) Classifier:
        # Non-parametric, instance-based classifier that computes Euclidean distance (Minkowski p=2):
        # d(x, x') = sqrt( sum_{j=1}^{D} (x_j - x'_j)^2 )
        # Predicts class via majority vote among the k=7 nearest neighbors in the training set.
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="uniform",
            metric="minkowski",
            p=2
        ),

        # 4. Gaussian Naive Bayes Classifier:
        # Generative classifier applying Bayes' Theorem under conditional feature independence:
        # P(y=k|x) proportional to P(y=k) * prod_{j=1}^{D} P(x_j | y=k)
        # Assumes each continuous feature follows a class-conditional Gaussian distribution:
        # P(x_j | y=k) = (1 / sqrt(2 * pi * sigma_{jk}^2)) * exp( -(x_j - mu_{jk})^2 / (2 * sigma_{jk}^2) )
        "Naive Bayes": GaussianNB(
            var_smoothing=1e-9
        ),

        # 5. Support Vector Classifier (SVM):
        # Finds the maximum-margin hyperplane solving the soft-margin dual optimization problem
        # with an RBF (Gaussian) kernel: K(x, x') = exp(-gamma * ||x - x'||_2^2).
        # C=2.0 controls the trade-off between margin width and hinge-loss slack penalties.
        # probability=True fits Platt Scaling (internal cross-validated sigmoid calibration)
        # so predict_proba() can output calibrated probabilities for ROC-AUC computation.
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=2.0,
            probability=True,
            random_state=42
        )
    }

    # =========================================================================
    # STEP 6: EVALUATE MODELS ON TEST SET & SAVE MODEL ARTIFACTS
    # =========================================================================
    evaluation_records = []

    for model_name, clf in models.items():
        # Train classifier on preprocessed training matrix
        clf.fit(X_train_processed, y_train)

        # Save trained model to 'model/' directory
        safe_filename = model_name.lower().replace(" ", "_") + "_model.joblib"
        joblib.dump(clf, os.path.join("model", safe_filename))

        # Generate discrete class predictions {0, 1} and positive-class probabilities P(y=1|x)
        y_pred = clf.predict(X_test_processed)
        y_prob = clf.predict_proba(X_test_processed)[:, 1]

        # ---------------------------------------------------------------------
        # Mathematical Definitions of the 6 Required Evaluation Metrics:
        # Let TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives
        # ---------------------------------------------------------------------

        # 1. Accuracy: Overall proportion of correct predictions across all instances
        # Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)
        acc = accuracy_score(y_test, y_pred)

        # 2. AUC Score (Area Under the Receiver Operating Characteristic Curve):
        # Integrates True Positive Rate vs. False Positive Rate across all decision thresholds:
        # AUC = integral_0^1 TPR(FPR) d(FPR) = P(Score(x_pos) > Score(x_neg))
        auc = roc_auc_score(y_test, y_prob)

        # 3. Precision (Positive Predictive Value):
        # Proportion of predicted passing students who actually passed
        # Formula: Precision = TP / (TP + FP)
        prec = precision_score(y_test, y_pred, zero_division=0)

        # 4. Recall (Sensitivity / True Positive Rate):
        # Proportion of actual passing students that the model correctly identified
        # Formula: Recall = TP / (TP + FN)
        rec = recall_score(y_test, y_pred, zero_division=0)

        # 5. F1 Score:
        # Harmonic mean of Precision and Recall, penalizing extreme imbalance between the two
        # Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # 6. Matthews Correlation Coefficient (MCC):
        # Pearson correlation coefficient between actual and predicted binary classifications.
        # Formula: MCC = (TP * TN - FP * FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
        # Ranges from -1 to +1; unlike Accuracy or F1, MCC only yields a high score if the
        # classifier performs well on both the majority (Pass) and minority (Fail) classes.
        mcc = matthews_corrcoef(y_test, y_pred)

        evaluation_records.append({
            "ML Model Name": model_name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4)
        })

    # =========================================================================
    # STEP 7: PRINT CLEAN COMPARISON TABLE
    # =========================================================================
    results_df = pd.DataFrame(evaluation_records)
    print("\n" + "=" * 82)
    print("BITS PILANI WILP M.TECH (AIML) - ML ASSIGNMENT 2 MODEL COMPARISON TABLE")
    print("=" * 82)
    print(results_df.to_string(index=False))
    print("=" * 82)
    print("[SUCCESS] All 5 models and preprocessor saved in 'model/' and 'test_data.csv' exported.")


if __name__ == "__main__":
    main()