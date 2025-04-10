
# 🧠 Oversampling Methods for Imbalanced Data in Credit Card Fraud Detection

This project investigates the effectiveness of various **oversampling techniques** for handling **imbalanced datasets** in the domain of **credit card fraud detection**, using both **machine learning** and **deep learning models**.


---

## 🎯 Objective

In fraud detection, the dataset is often highly **imbalanced**, where legitimate transactions vastly outnumber fraudulent ones. This project evaluates how well oversampling methods such as **RandomOverSampler**, **SMOTE**, and **ADASYN** improve model performance on such data.

---

## 📊 Dataset

**Source**:  
📥 [Credit Fraud - Dealing with Imbalanced Datasets](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/input)

- **Total Records**: 284,807  
- **Fraud Cases**: 492 (~0.17%)  
- **Features**: 30 (including PCA-transformed `V1` to `V28`, `Time`, `Amount`, and `Class`)

> ⚠️ To run the code, download `creditcard.csv` from the Kaggle link above and place it in the project root folder.  
> It is not included in this repository due to GitHub's 100MB file limit.

---

## 🧪 Oversampling Techniques

| Method            | Description                                                        |
|------------------|--------------------------------------------------------------------|
| RandomOverSampler| Randomly duplicates minority class samples                         |
| SMOTE            | Creates synthetic samples between existing minority instances      |
| ADASYN           | Creates more synthetic samples for harder-to-classify minority points|

---

## 🤖 Models Used

### 🔷 Machine Learning
- Random Forest Classifier
- Logistic Regression
- Decision Tree Classifier

### 🔶 Deep Learning
- Artificial Neural Network (ANN)
- Recurrent Neural Network (RNN)

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

> Special focus is placed on **Recall** and **F1-Score** to better evaluate fraud detection performance on imbalanced data.

---

## 🖼️ Visualizations

- Log-transformed distribution of transaction amounts
- Pie chart of top 5 active transaction time segments
- Bar plots of class-wise transaction amounts
- Correlation heatmaps
- Model performance comparisons
- Confusion matrices

---

## 📋 Results Summary

| Model               | Accuracy (%) | Notes                                 |
|--------------------|--------------|----------------------------------------|
| Random Forest       | 99.99        | Highest performing ML model            |
| Logistic Regression | 97.11        | Fast and interpretable                 |
| Decision Tree       | 99.83        | Simple but highly accurate             |
| ANN                 | 97.10        | Effective deep learning model          |
| RNN                 | 96.63        | Lower than ANN, sequential modeling    |

> **SMOTE** consistently improved F1 and Recall across models.

---

## 🧰 Tech Stack

- Python 3.9+
- PySpark
- Scikit-learn
- TensorFlow / Keras
- imbalanced-learn
- Pandas, NumPy
- Matplotlib, Seaborn

---

