# 🚀 Quick Start Guide

## Comprehensive Stroke Prediction with SMOTE-NC, MLP, and Statistical Analysis

### What's New? ✨

This project now includes:
- ✅ **MLP (Neural Network)** instead of XGBoost
- ✅ **SMOTE-NC** comparison (with/without oversampling)
- ✅ **10-Fold Cross-Validation**
- ✅ **Paired t-test** statistical analysis
- ✅ **Professional Streamlit GUI**

---

## Step-by-Step Guide

### Step 1: Install Dependencies ⚙️

```bash
pip install -r requirements.txt
```

This installs:
- scikit-learn (machine learning)
- imbalanced-learn (SMOTE-NC)
- streamlit (GUI)
- scipy (statistical tests)
- plotly, matplotlib, seaborn (visualizations)

---

### Step 2: Train All Models 🎓

```bash
python train_models_comprehensive.py
```

**What happens:**
1. Loads dataset (5,110 patients)
2. Applies SMOTE-NC to balance classes
3. Trains 4 models (Logistic Regression, Decision Tree, Random Forest, **MLP**)
4. Tests each model WITH and WITHOUT SMOTE-NC
5. Performs 10-fold cross-validation
6. Runs paired t-tests
7. Saves best model

**Expected runtime:** 2-5 minutes

**Generated files:**
- `best_stroke_model.pkl`
- `scaler.pkl`
- `model_comparison_results.csv`
- `kfold_cv_detailed_results.csv`

---

### Step 3: Generate Visualizations 📊

```bash
python generate_visualizations.py
```

**Creates 6 publication-ready figures:**
1. Model comparison bar charts
2. K-Fold CV box plots
3. Performance heatmaps
4. F1-Score across folds
5. SMOTE-NC improvement chart
6. Summary statistics table

**Output:** PNG files ready for your report!

---

### Step 4: Launch GUI Application 🖥️

```bash
streamlit run app.py
```

**Features:**
- Enter patient data via web interface
- Get real-time stroke risk prediction
- View risk visualization (gauge chart)
- Receive personalized recommendations
- Professional medical-style UI

**Access:** Opens automatically in browser (usually `http://localhost:8501`)

---

## Understanding Your Results

### Console Output Explanation

```
🔬 MLP:
   Mean F1 (No SMOTE):   0.2845 ± 0.0312
   Mean F1 (SMOTE-NC):   0.3421 ± 0.0289
   Difference:           +0.0576
   t-statistic:          3.8421
   p-value:              0.0034 **
   ✓ SMOTE-NC is significantly better (p < 0.05)
```

**Interpretation:**
- MLP with SMOTE-NC achieves 34.21% F1-score
- This is 5.76 percentage points better than without SMOTE
- p-value < 0.05 → statistically significant improvement
- `**` indicates high confidence (p < 0.01)

---

## Key Features Explained

### 🧪 SMOTE-NC (Synthetic Minority Over-sampling)

**Problem:** Dataset has 20× more non-stroke cases than stroke cases

**Solution:** SMOTE-NC creates synthetic stroke cases for training
- Handles both categorical (gender, smoking) and continuous (age, BMI) features
- Only applied to training data (NOT test data)
- Balances classes to help model learn stroke patterns

**Result:** Better recall (finds more stroke cases)

---

### 🧠 MLP (Multi-Layer Perceptron)

**Architecture:**
```
Input (10 features)
    ↓
Hidden Layer 1 (100 neurons, ReLU)
    ↓
Hidden Layer 2 (50 neurons, ReLU)
    ↓
Hidden Layer 3 (25 neurons, ReLU)
    ↓
Output (2 classes: stroke/no stroke)
```

**Why MLP?**
- Captures complex non-linear patterns
- Better than tree-based models for this dataset
- Proven effectiveness in medical diagnosis

---

### 📈 K-Fold Cross-Validation (K=10)

**What it does:**
1. Splits data into 10 parts
2. Trains on 9 parts, tests on 1 part
3. Repeats 10 times (each part used as test once)
4. Reports average performance

**Why important:**
- More reliable than single train/test split
- Reduces overfitting
- Shows model stability

---

### 📊 Paired t-test

**Purpose:** Statistical comparison between models

**Example:**
```
MLP vs Random Forest:
p-value = 0.0123 * → MLP is significantly better
```

**Significance levels:**
- `***` p < 0.001 → Very strong evidence
- `**` p < 0.01 → Strong evidence
- `*` p < 0.05 → Significant
- `ns` → No significant difference

---

## Troubleshooting 🔧

### ❌ Error: Module not found
```bash
pip install --upgrade -r requirements.txt
```

### ❌ Error: Model files not found
```bash
python train_models_comprehensive.py
```
(Must train models before running GUI)

### ❌ Error: Cannot generate visualizations
Make sure you've run training script first to generate CSV files

### ⚠️ Warning: MLP training is slow
Normal! Neural networks take longer. Wait 2-5 minutes.

---

## For Your Report 📝

### Include These Results:

1. **Comparison Table** (from CSV)
   - Shows all models with/without SMOTE-NC
   - F1, Recall, Precision, ROC-AUC scores

2. **Statistical Analysis**
   - Paired t-test results
   - Significance levels
   - Confidence intervals

3. **Visualizations** (6 figures)
   - Bar charts, box plots, heatmaps
   - Professional quality (300 DPI)

4. **Best Model**
   - Which model performed best?
   - With or without SMOTE-NC?
   - Why?

5. **Feature Importance**
   - Which factors most predict stroke?
   - Clinical interpretation

---

## Project Timeline ⏱️

1. **Data Preparation** ✅ (Already done)
2. **Model Training** ⏱️ (Run `train_models_comprehensive.py`)
3. **Results Analysis** ⏱️ (Run `generate_visualizations.py`)
4. **GUI Demo** ⏱️ (Run `streamlit run app.py`)
5. **Final Report** 📝 (Use generated results)

---

## Tips for Presentation 🎤

### Demo Workflow:

1. **Show Code Structure**
   - Explain SMOTE-NC implementation
   - Show MLP architecture
   - Discuss cross-validation

2. **Present Results**
   - Display comparison table
   - Explain statistical significance
   - Show visualizations

3. **Live Demo**
   - Open Streamlit app
   - Enter sample patient data
   - Show prediction + explanation

4. **Discuss Findings**
   - Which model is best? (Likely MLP with SMOTE-NC)
   - Why does SMOTE-NC help?
   - Clinical implications

---

## Sample Patient Data for Demo 👤

**High Risk Patient:**
- Age: 75
- Hypertension: Yes
- Heart Disease: Yes
- Glucose: 200 mg/dL
- BMI: 32
- Smoking: Smokes
- **Expected:** High risk (>60%)

**Low Risk Patient:**
- Age: 25
- Hypertension: No
- Heart Disease: No
- Glucose: 90 mg/dL
- BMI: 22
- Smoking: Never smoked
- **Expected:** Low risk (<20%)

---

## Files Overview 📁

### Scripts:
- `train_models_comprehensive.py` - Main training (NEW!)
- `generate_visualizations.py` - Create plots (NEW!)
- `app.py` - Streamlit GUI
- `data_factory/data_factory.py` - Data preprocessing

### Generated:
- `best_stroke_model.pkl` - Trained model
- `scaler.pkl` - Data normalizer
- `*.csv` - Results tables
- `figure*.png` - Visualizations

### Documentation:
- `TRAINING_GUIDE.md` - Detailed guide
- `QUICK_START.md` - This file
- `README.md` - Project overview

---

## Need Help? 💬

**Common Questions:**

**Q: Which model is best?**
A: Run training script - it automatically selects and saves the best model based on F1-score.

**Q: Why use F1-score instead of accuracy?**
A: Dataset is imbalanced (20:1). Accuracy would be misleading. F1-score balances precision and recall.

**Q: Should I always use SMOTE-NC?**
A: For training, yes (helps with imbalance). For testing, NO (test on real data).

**Q: How do I tune hyperparameters?**
A: Edit model configurations in `train_models_comprehensive.py` and re-run.

---

## Success Checklist ✅

Before your presentation:

- [ ] Installed all dependencies
- [ ] Trained all models successfully
- [ ] Generated visualizations
- [ ] Tested GUI application
- [ ] Reviewed comparison results
- [ ] Understood statistical tests
- [ ] Prepared demo patient data
- [ ] Created presentation slides

---

## Authors

👥 **Team Members:**
- Tugan Başaran
- Zehra Sağın
- Mert Korkmaz

📅 **Date:** January 2026

🎓 **Project:** Predicting Stroke Occurrence Using Medical Patient Data and Machine Learning

---

**Good luck with your presentation! 🎉**
