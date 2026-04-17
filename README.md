# Comment-Category-Prediction-using-ML
In this competition, I analyzed a dataset containing information about short textual entries submitted to an online discussion system. The dataset includes the content of each entry, engagement signals from other users, and outputs from automated analysis components.

# 🧠 Comment Category Prediction (Kaggle Project)

📊 **Best Model:** Logistic Regression 
🏆 **Macro F1 Score:** 0.8057 
📁 **Dataset Size:** ~198K rows  

---

## 📌 Problem Statement

The goal of this project is to classify user-generated comments into **4 categories (0,1,2,3)** using:

- 📝 Text data (comments)
- 📊 Numerical metadata (votes, interactions, etc.)

### ⚠️ Challenge:
- Severe **class imbalance**
- Mixed data types (text + numerical)
- Evaluation metric: **Macro F1 Score**

---

## 📂 Dataset

Due to GitHub file size limits, datasets are hosted on Google Drive:

- 📥 **Train Dataset:** [https://drive.google.com/file/d/1yNsWB0VH_U5AkaVQutyleSRcJ2zkRr80/view?usp=sharing]
- 📥 **Test Dataset:** [https://drive.google.com/file/d/1l1Yi307QVQmokN1ETJ2Gu6zSS8FkhpBJ/view?usp=sharing]
- 📥 **Sample Submission:** [https://drive.google.com/file/d/1pvtPYqIzV4fU9JgfbgnXIeoRX98reZu_/view?usp=sharing]

---

## ⚙️ Tech Stack

- Python 🐍
- pandas, numpy
- scikit-learn
- LightGBM
- seaborn, matplotlib

---

## 🔍 Key Steps

### 1. 📊 Exploratory Data Analysis (EDA)
- Identified **severe class imbalance**
- handled high-missing columns (`race`, `religion`, `gender`),approximately 436k missing values.
- Found strong signals in:
  - emoticons and downvote/upvote
- Visual Aids using histograms,boxplots and pairplots.
---

### 2. 🧹 Feature Engineering

#### 🔢 Numerical Features
- Extracting Features:
  - 'created_date' feature used for extracting features like year,month,day,dayofweek,weekend,hour etc
  

#### 🕒 Temporal Features
- Extracted:
  - `hour`, `dayofweek`, `month`

#### 📝 Text Processing
- Cleaning:
  - Removed URLs, mentions, hashtags
  - Lowercasing & regex filtering
- Features:
  - **Word TF-IDF (1–3 grams)**
  - **Character TF-IDF (2–5 grams)**

---

### 3.)🤖 Models Used

| Model | Macro F1 | Rank |
|------|--------|------|
| Logistic Regression | **0.8057** | 🥇 |
| LightGBM | 0.78 | 🥈 |
| XGBoost | 0.76 | 🥉 |

---

## 🥇 Best Model: Logistic Regression

### Why it worked best:

- Handles **linear feature interactions**
- Combines:
  - TF-IDF text features
  - Engineered numerical features
- Robust to **outliers**
- Handles **class imbalance effectively**

---

## 📈 Key Insights

- 📌 `if_2_log` → strongest predictor for class 0  
- 📌 `if_1_log` → strong separator for minority class  
- 📌 Text features dominate, but **numerical features boosted performance**
- 📌 Class imbalance handling was **critical**

---

## ⚠️ Challenges

- Severe imbalance (55% vs 4%)
- Missing demographic features (~73%)
- Validation vs test performance gap
- Limited hyperparameter search

---

## 🚀 Future Improvements

- 🔥 Transformer models (BERT / RoBERTa)
- 🔗 Model stacking (LightGBM + Logistic Regression)
- 🎯 Threshold tuning per class
- 🔁 Cross-validation for better generalization

---
