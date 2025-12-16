# Predicting Stroke Occurrence Using Medical Patient Data and Machine Learning

**Group Members:**

- S033995 Tugan Başaran
- S033273 Zehra Sağın
- S034587 Mert Korkmaz

---

## 1. Problem Description

Stroke is one of the leading global causes of death and early identification is crucial. In this project we aim to use and compare three different algorithms to predict stroke likelihood based on clinical features. We also focus on the interpretability of predictions, investigating how clinical features such as age, gender, heart disease have an effect. By using algorithms such as decision trees, we aim to understand the key drivers of stroke occurrence alongside high predictive performance.

---

## 2. Dataset

We will use the **Stroke Prediction Dataset**, which is available on Kaggle and sourced from confidential real patient records.

**Link:** [https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

### Dataset Overview:

- **Size:** 5,110 instances.
- **Type:** Real-world medical data.
- **Target Variable:** `Stroke` (1 = Yes, 0 = No).
- **Features:** 11 clinical features.

The dataset includes a mix of categorical and numerical features representing demographic, lifestyle, and clinical health attributes:

- **Demographic:** Gender, Age, Ever Married, Residence Type, Work Type.
- **Clinical/Physiological:** Hypertension, Heart Disease, Average Glucose Level, BMI.
- **Lifestyle:** Smoking Status.

The dataset will be split as followed:

- 70% Training
- 15% Validation
- 15% Testing

---

## 3. Proposed Approach

### 3.1 Algorithms

In line with the objective of understanding the decision-making process, we will prioritize algorithms that offer interpretability alongside those known for high performance. We will evaluate the following classification algorithms:

1.  **Decision Tree Classifier**
2.  **Logistic Regression**
3.  **Random Forest**
4.  **XGBoost**

### 3.2 Evaluation Metrics

Since this is a medical diagnosis problem with an imbalanced dataset (fewer stroke cases compared to no-stroke cases), standard accuracy alone is insufficient. We will use:

- **Recall (Sensitivity)**
- **Precision & F1-Score**
- **ROC-AUC**
- **Confusion Matrix**
- **Feature Importance Scores:** Specifically to address the research question regarding which inputs (Age, Ever Married, etc.) have the most effect.

---

## 4. Related Work

In medical informatics, stroke prediction has been widely studied. Although methods such as Deep Learning and Support Vector Machines (SVM) are widely used, recent reports emphasized the importance of Explainable AI (XAI), pointing out that black-box models are not suitable for clinical decision support. Our study is consistent with other research that uses rule-based models and decision trees to connect demographic factors and living environment to health outcomes. We employ a similar methodology to analyses on the Framingham Heart Study utilizing the Kaggle Stroke Prediction dataset, concentrating on deriving significant health insights from tabular patient data.

---

## References

1.  Fedesoriano. (2021). _Stroke prediction dataset_. [https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
2.  Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). _Classification and regression trees_. Wadsworth.
3.  World Health Organization. (2020). _Global health estimates: Life expectancy and leading causes of death and disability_.
