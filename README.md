# Machine Learning Assignment - 2: Student Academic Performance Classification

**Course:** Machine Learning (M.Tech AIML — BITS Pilani WILP)  
**Assignment:** Assignment - 2 (End-to-End Classification & Streamlit Cloud Deployment)

---

## a. Problem Statement

Early identification of students at risk of academic failure enables educators to provide targeted academic support and counseling before final examinations occur. The objective of this project is to build, evaluate, and deploy an end-to-end supervised binary classification pipeline that predicts whether a secondary school student will **Pass (`1`, where final grade `G3 >= 10`)** or **Fail (`0`, where final grade `G3 < 10`)** their Portuguese language course using socio-demographic, familial, lifestyle, and prior academic engagement attributes.

To simulate a realistic early-warning scenario and prevent target leakage, first-period (`G1`) and second-period (`G2`) interim grades are excluded from the feature set, ensuring the classifiers learn purely from behavioral and demographic predictors.

---

## b. Dataset Description

- **Dataset Name:** UCI Student Performance Dataset (`student-por.csv` — Portuguese Language Course)
- **Source Repository:** UCI Machine Learning Repository (Cortez & Silva, 2008)
- **Number of Instances:** `649` students (exceeds the assignment minimum requirement of 500 instances)
- **Number of Raw Attributes:** `33` attributes (`30` socio-demographic/behavioral input features after dropping `G1`, `G2`, and `G3` to prevent data leakage; exceeds the minimum requirement of 12 features)
- **Target Variable (`pass`):** Binary classification label derived from final grade `G3` (scaled 0–20):
  - `1 (Pass)`: `G3 >= 10` (`549` instances, `84.59%`)
  - `0 (Fail)`: `G3 < 10` (`100` instances, `15.41%`)
- **Feature Breakdown:**
  - **Numerical Features (13):** `age`, `Medu` (mother's education), `Fedu` (father's education), `traveltime`, `studytime`, `failures` (past class failures), `famrel` (family relationship quality), `freetime`, `goout`, `Dalc` (workday alcohol consumption), `Walc` (weekend alcohol consumption), `health`, and `absences`. Standardized using `StandardScaler` fit strictly on the training split (`80%`, `519` rows).
  - **Categorical Features (17):** `school`, `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, and `romantic`. Encoded using `OneHotEncoder(handle_unknown='ignore')` fit strictly on the training split.

---

## c. GitHub Repository Link

- **GitHub Repository URL:** [https://github.com/Abijith-2026aj05062/bits-wilp-ml-assignment-2](https://github.com/Abijith-2026aj05062/bits-wilp-ml-assignment-2)
- **Live Streamlit Cloud App URL:** [https://bits-wilp-ml-assignment-2-dxgbffvv9vq5canikzduvz.streamlit.app/](https://bits-wilp-ml-assignment-2-dxgbffvv9vq5canikzduvz.streamlit.app/)

---

## d. Models Used

### 1. Model Evaluation Comparison Table (Test Split: `N = 130`)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8769 | 0.7930 | 0.9160 | 0.9478 | 0.9316 | 0.3228 |
| **Decision Tree** | 0.8692 | 0.6438 | 0.9153 | 0.9391 | 0.9270 | 0.3007 |
| **KNN** | 0.8846 | 0.7499 | 0.8968 | 0.9826 | 0.9378 | 0.2145 |
| **Naive Bayes** | 0.8231 | 0.7339 | 0.9340 | 0.8609 | 0.8959 | 0.3246 |
| **Support Vector Machine** | **0.9000** | **0.8017** | 0.9113 | **0.9826** | **0.9456** | **0.3796** |

---

### 2. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Delivered highly balanced discriminative performance (`Accuracy = 0.8769`, `AUC = 0.7930`, `F1 = 0.9316`, `MCC = 0.3228`). Because `OneHotEncoder` expands the categorical predictors into a 43-dimensional feature space, L2-regularized linear log-odds effectively captured additive relationships (such as `failures`, `higher` education aspiration, and `studytime`) without overfitting. |
| **Decision Tree** | Achieved respectable accuracy (`0.8692`) and F1 (`0.9270`) with `max_depth=5` pre-pruning, but recorded the lowest `AUC` (`0.6438`) among all five models. Axis-aligned greedy splits on sparse one-hot encoded categorical variables produce coarse piecewise-constant probability estimates at leaf nodes, degrading rank-ordering capability across thresholds. |
| **KNN** | Reached high overall accuracy (`0.8846`) and F1 (`0.9378`) driven by high recall (`0.9826`), but yielded the lowest `MCC` (`0.2145`). In a 43-dimensional feature space with an `84.6%` majority pass rate, uniform Euclidean neighborhoods (`k=7`) suffer from the curse of dimensionality and bias local majority voting toward the dominant `Pass` class, causing false positives on failing students. |
| **Naive Bayes** | Achieved the highest `Precision` (`0.9340`) and the second-highest `MCC` (`0.3246`), aggressive at flagging at-risk `Fail` students, which lowered `Recall` (`0.8609`) and overall `Accuracy` (`0.8231`). This behavior stems from two factors: one-hot encoded binary indicators violate the Gaussian likelihood assumption, and correlated socio-economic variables (`Medu`, `Fedu`, `Mjob`, `higher`) violate conditional independence, pushing posterior probabilities toward extremes. |
| **Support Vector Machine** | Outperformed all competing algorithms across **Accuracy (`0.9000`)**, **AUC (`0.8017`)**, **Recall (`0.9826`)**, **F1 (`0.9456`)**, and **MCC (`0.3796`)**. The Radial Basis Function (RBF) kernel effectively mapped non-linear interactions between lifestyle habits (such as `Dalc`/`Walc`, `goout`, `absences`) and academic history (`failures`) while controlling margin violations via `C=2.0`. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVC with RBF Kernel)** is the clear overall winner for this dataset. Because the dataset exhibits class imbalance (`84.6%` Pass vs. `15.4%` Fail), relying on Accuracy alone can be misleading. SVM simultaneously achieves the highest **MCC (`0.3796`)**, **ROC-AUC (`0.8017`)**, **F1 Score (`0.9456`)**, and **Accuracy (`0.9000`)**, proving that it generalizes best across both passing and failing student cohorts. **Logistic Regression** is the runner-up (`AUC = 0.7930`, `MCC = 0.3228`), offering strong linear interpretability with only a slight trade-off in non-linear boundary capacity. |