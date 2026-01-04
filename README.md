# AI-Driven Predictive Maintenance for Manufacturing Equipment

## 📌 Overview

This project builds a machine learning system to **predict industrial equipment failures before they occur**, using the **AI4I 2020 Predictive Maintenance Dataset**.  
The goal is to help manufacturing plants schedule **preventive maintenance**, reduce unplanned downtime, and avoid costly breakdowns.

---

## 🏭 Problem Statement

Industrial machines often fail unexpectedly, causing:

- Production stoppage  
- Emergency repair costs  
- Quality issues in finished products  

By predicting failures 1–2 days in advance, maintenance teams can:

- Plan repairs during planned shutdowns  
- Order spare parts early  
- Extend equipment lifetime  

This project answers the question:

> **“Given the current sensor readings of a machine, will it fail soon?”**

---

## 📂 Dataset

- **Name:** AI4I 2020 Predictive Maintenance Dataset  
- **Records:** 10,000 machine operations  
- **Target:** `Machine failure` (0 = No failure, 1 = Failure)  
- **Original class balance:**  
  - ~97% No failure  
  - ~3% Failure (highly imbalanced)

**Key sensor features:**

- Air temperature  
- Process temperature  
- Rotational speed (RPM)  
- Torque  
- Tool wear (minutes)  
- Quality type (H, M, L)

---

## 🧹 Data Preparation

Steps performed in the notebook:

1. **Data loading**
   - Loaded `ai4i2020.csv` into a pandas DataFrame.

2. **Data cleaning**
   - Checked for missing values.  
   - Checked for duplicate rows.  
   - Cleaned column names (removed spaces and brackets).

3. **Feature engineering**
   Created domain-specific features to capture stress on the machine:

   - `Temperature_Ratio` = Process temperature / Air temperature  
   - `Power_Indicator` = Rotational speed × Torque  
   - `Tool_Wear_Squared` = \(\text{Tool wear}^2\)  
   - `Stress_Score` = Combined score using temperature, torque, speed, and tool wear  

4. **Scaling**
   - Scaled numeric features to **0–1 range** using Min-Max scaling.

5. **Class balancing (SMOTE)**
   - Original data:  
     - ~9,661 “no failure”  
     - ~339 “failure”  
   - Applied **SMOTE** to create synthetic failure samples.  
   - After SMOTE: **50% no failure, 50% failure** (perfectly balanced).

6. **Train–test split**
   - 80% training data  
   - 20% test data  
   - Both sets remain balanced after SMOTE.

---

## 🤖 Models Trained

Three classification models were trained and compared:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Machine (SVM)**

For each model, the following metrics were calculated on the test set:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC score  

---

## 📈 Results (Example)

> These are example numbers to show how results are summarized; your exact values may differ slightly depending on random seeds and feature choices.

### Best Model: **Random Forest**

| Metric     | Value (approx) |
|-----------|-----------------|
| Accuracy  | ~91%            |
| Precision | ~89%            |
| Recall    | ~88%            |
| F1-score  | ~0.88           |
| ROC-AUC   | ~0.95           |

### Other Models

- **Logistic Regression:** ~82% accuracy  
- **SVM:** ~88% accuracy  

Random Forest was selected as the **final model** because it provided the best balance of accuracy, precision, and recall.

---

## 🌟 Feature Importance (Random Forest)

Top features influencing failure prediction:

1. **Tool wear** – Most important predictor of failure  
2. **Process temperature** – Higher values increase risk  
3. **Torque** – High load increases stress on the machine  
4. **Rotational speed (RPM)**  
5. **Air temperature**  

These align with real-world intuition: worn tools, high temperature, and heavy load all increase the chance of failure.

---

## 📊 Visualizations

The notebook includes several plots:

- Failure distribution (class imbalance)  
- Histograms of tool wear and temperature  
- Scatter plots:
  - Temperature vs failure  
  - Torque vs failure  
- Quality type vs failure count  
- Confusion matrix heatmap for the best model  
- Model comparison bar charts (accuracy and F1-score)  
- Feature importance bar chart  
- ROC curve for the best model  

These help explain **why** the model behaves the way it does.

---

## 🏭 Business Value

Using this predictive model, a manufacturing plant can:

- Detect high-risk machines **before** failure  
- Schedule maintenance in advance  
- Reduce unplanned downtime and emergency repairs  
- Improve safety and product quality  

Even a small improvement in failure prediction can save **significant costs** at industrial scale.

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:**  
  - pandas, numpy  
  - scikit-learn  
  - imbalanced-learn (SMOTE)  
  - matplotlib, seaborn  

---

## 🚀 How to Run

### Option 1: Google Colab

1. Open the notebook in Google Colab.  
2. Upload `ai4i2020.csv`.  
3. Run all cells from top to bottom.  

### Option 2: Local Jupyter

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
jupyter notebook
