# Comprehensive Training Guide

## Overview
This project implements a comprehensive stroke prediction system with:
- ✅ **MLP (Multi-Layer Perceptron)** neural network model
- ✅ **K-Fold Cross-Validation** (10 folds)
- ✅ **SMOTE-NC** comparison (with and without oversampling)
- ✅ **Paired t-test** statistical analysis
- ✅ **GUI with Streamlit** for real-time predictions

## Authors
- Tugan Başaran
- Zehra Sağın  
- Mert Korkmaz

---

## Installation

### 1. Install Required Packages
```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- scikit-learn
- numpy
- matplotlib
- seaborn
- scipy
- streamlit
- imbalanced-learn
- plotly
- joblib

---

## Training the Models

### Run Comprehensive Training Script

```bash
python train_models_comprehensive.py
```

This script will:

1. **Load and prepare the dataset** (70% train, 15% validation, 15% test)
2. **Apply SMOTE-NC** to handle class imbalance
3. **Train 4 models** both WITH and WITHOUT SMOTE-NC:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - **MLP (Multi-Layer Perceptron)** - Neural Network
4. **Perform 10-Fold Cross-Validation** for each model
5. **Run Paired t-tests** to statistically compare:
   - Models with SMOTE-NC vs without SMOTE-NC
   - MLP vs other models
6. **Select and save the best model** for deployment

### Expected Output

The script will generate:

#### Console Output:
- Dataset information and class distribution
- SMOTE-NC application results
- K-Fold CV results for each model (Mean ± Std)
- Test set performance metrics
- Comparison table (SMOTE vs No SMOTE)
- Paired t-test results with significance levels
- Confusion matrices
- Best model selection

#### Generated Files:
- `best_stroke_model.pkl` - Best performing model
- `scaler.pkl` - StandardScaler for input normalization
- `model_comparison_results.csv` - Summary comparison table
- `kfold_cv_detailed_results.csv` - Detailed CV results for all folds

---

## Understanding the Results

### Metrics Used

1. **F1-Score**: Harmonic mean of precision and recall (main metric for imbalanced data)
2. **Recall**: Ability to find all stroke cases (important for medical diagnosis)
3. **Precision**: Accuracy of stroke predictions
4. **ROC-AUC**: Area Under the ROC Curve (overall model performance)

### K-Fold Cross-Validation

- Uses **Stratified K-Fold** with K=10
- Preserves class distribution in each fold
- Reports mean and standard deviation across folds
- Provides robust performance estimates

### SMOTE-NC (Synthetic Minority Over-sampling Technique)

**What it does:**
- Creates synthetic samples for the minority class (stroke cases)
- Handles both categorical and continuous features
- Balances the training dataset

**Categorical features in dataset:**
- Gender, Hypertension, Heart Disease, Ever Married, Work Type, Residence Type, Smoking Status

**When to use:**
- Training phase only
- NOT applied to validation/test sets
- Helps models learn stroke patterns better

### Paired t-test Analysis

**Purpose:** Statistical comparison between models

**Null Hypothesis (H0):** No difference between models

**Significance levels:**
- `***` p < 0.001 (highly significant)
- `**` p < 0.01 (very significant)
- `*` p < 0.05 (significant)
- `ns` not significant

**Interpretation:**
- If p < 0.05: Reject H0, models are significantly different
- If p ≥ 0.05: Cannot reject H0, no significant difference

---

## Running the GUI Application

After training, launch the Streamlit app:

```bash
streamlit run app.py
```

### Features:
- Interactive patient data input
- Real-time stroke risk prediction
- Risk visualization with gauge chart
- Risk factor analysis
- Medical recommendations
- Professional medical-style interface

### Using the App:
1. Fill in patient information in the sidebar:
   - Demographics (age, gender, marital status, etc.)
   - Clinical data (hypertension, heart disease, glucose, BMI)
   - Lifestyle (smoking status)
2. Click "Predict Stroke Risk"
3. View results:
   - Risk level (Low/Medium/High)
   - Probability percentage
   - Risk factor analysis
   - Medical recommendations

---

## Model Architecture

### MLP (Multi-Layer Perceptron) Configuration
```python
MLPClassifier(
    hidden_layers=(100, 50, 25),  # 3 hidden layers
    activation='relu',             # ReLU activation
    solver='adam',                 # Adam optimizer
    max_iter=500,                  # Training epochs
    early_stopping=True,           # Prevent overfitting
    validation_fraction=0.15       # Internal validation
)
```

**Architecture:**
- Input Layer: 10 features
- Hidden Layer 1: 100 neurons + ReLU
- Hidden Layer 2: 50 neurons + ReLU  
- Hidden Layer 3: 25 neurons + ReLU
- Output Layer: 2 classes (stroke/no stroke) + Softmax

---

## Dataset Information

**Source:** Stroke Prediction Dataset (Kaggle)

**Features (10):**
1. Gender (categorical)
2. Age (continuous)
3. Hypertension (binary)
4. Heart Disease (binary)
5. Ever Married (binary)
6. Work Type (categorical)
7. Residence Type (binary)
8. Average Glucose Level (continuous)
9. BMI (continuous)
10. Smoking Status (categorical)

**Target:** Stroke (0 = No, 1 = Yes)

**Class Imbalance:** ~20:1 ratio (No Stroke : Stroke)

---

## Troubleshooting

### Issue: Model files not found
**Solution:** Run `python train_models_comprehensive.py` first

### Issue: Package import errors
**Solution:** 
```bash
pip install --upgrade -r requirements.txt
```

### Issue: SMOTE-NC fails
**Solution:** Check that categorical feature indices match your data encoding

### Issue: MLP training is slow
**Solution:** 
- Reduce `max_iter` parameter
- Use fewer neurons in hidden layers
- Enable GPU if available (requires additional setup)

---

## Tips for Better Results

### Hyperparameter Tuning:
1. **MLP:** Adjust hidden layer sizes, learning rate, activation functions
2. **Random Forest:** Tune n_estimators, max_depth, min_samples_split
3. **Logistic Regression:** Experiment with C parameter, penalty types

### Feature Engineering:
- Create interaction features (e.g., age × hypertension)
- Polynomial features for non-linear relationships
- Feature selection using importance scores

### Threshold Tuning:
- Adjust decision threshold based on use case
- For medical diagnosis: prioritize recall (find all stroke cases)
- Balance false positives vs false negatives

---

## Results Interpretation Example

```
🔬 MLP vs Random Forest:
   MLP F1:               0.3245 ± 0.0421
   Random Forest F1:     0.2891 ± 0.0389
   Difference:           +0.0354
   p-value:              0.0123 *
   ✓ MLP performs significantly better
```

**Interpretation:**
- MLP achieves 32.45% F1-score on average
- Random Forest achieves 28.91% F1-score
- MLP is 3.54 percentage points better
- p-value = 0.0123 < 0.05 → statistically significant
- Conclusion: MLP is significantly better than Random Forest

---

## Project Structure

```
├── data/
│   └── data.csv                          # Dataset
├── data_factory/
│   └── data_factory.py                   # Data preprocessing
├── train_models_comprehensive.py         # Main training script (NEW)
├── run_main.py                           # Original training script
├── app.py                                # Streamlit GUI
├── requirements.txt                      # Dependencies
├── TRAINING_GUIDE.md                     # This file
├── README.md                             # Project overview
├── best_stroke_model.pkl                 # Saved model (generated)
├── scaler.pkl                            # Saved scaler (generated)
├── model_comparison_results.csv          # Results (generated)
└── kfold_cv_detailed_results.csv         # Detailed CV (generated)
```

---

## Next Steps for Final Report

### 1. Results Analysis
- Compare all models with tables and charts
- Analyze confusion matrices
- Discuss SMOTE-NC impact
- Interpret statistical tests

### 2. Feature Importance
- Extract feature importance from Random Forest
- Analyze MLP weights (optional)
- Discuss clinical implications

### 3. Limitations
- Class imbalance challenges
- Limited dataset size
- Generalization to other populations

### 4. Future Work
- Ensemble methods combining multiple models
- Additional features (family history, medications)
- Real-time deployment considerations
- Integration with hospital systems

---

## Contact

For questions or issues:
- Tugan Başaran
- Zehra Sağın
- Mert Korkmaz

---

## References

1. SMOTE-NC: Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique"
2. K-Fold CV: Kohavi, "A Study of Cross-Validation and Bootstrap"
3. Paired t-test: Student's t-test for dependent samples
4. MLP: Rumelhart et al., "Learning Internal Representations by Error Propagation"

---

**Good luck with your project! 🎓**
