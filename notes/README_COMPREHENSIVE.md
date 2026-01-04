# 🏥 Predicting Stroke Occurrence Using Medical Patient Data

**A Comprehensive Machine Learning Project with SMOTE-NC, MLP, K-Fold CV, and Statistical Analysis**

---

## 👥 Group Members
- **S033995 Tugan Başaran**
- **S033273 Zehra Sağın**
- **S034587 Mert Korkmaz**

**Date:** January 2026

---

## 🎯 Project Overview

This project implements a comprehensive machine learning system to predict stroke occurrence in patients based on medical and demographic data. The system addresses class imbalance using SMOTE-NC, evaluates multiple models with K-Fold Cross-Validation, performs rigorous statistical testing, and provides a professional GUI for real-time predictions.

---

## ✨ Key Features

### Implemented Techniques:
- ✅ **SMOTE-NC** - Synthetic Minority Over-sampling for mixed data types
- ✅ **MLP Neural Network** - Multi-Layer Perceptron (replaced XGBoost)
- ✅ **K-Fold Cross-Validation** - 10-fold stratified validation
- ✅ **Paired t-test** - Statistical significance testing
- ✅ **Streamlit GUI** - Professional web interface for predictions
- ✅ **Comprehensive Comparison** - Each model tested WITH and WITHOUT SMOTE-NC

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train All Models
```bash
python train_models_comprehensive.py
```
*Runtime: 2-5 minutes | Output: Trained models, CSV results, statistical analysis*

### 3. Generate Visualizations
```bash
python generate_visualizations.py
```
*Output: 6 publication-ready figures (PNG)*

### 4. Launch GUI Application
```bash
streamlit run app.py
```
*Access: http://localhost:8501*

---

## 📊 Dataset

**Source:** Stroke Prediction Dataset from Kaggle  
**Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

### Dataset Statistics:
- **Total Samples:** 5,110 patients
- **Features:** 10 (mixed categorical/continuous)
- **Target:** Stroke (0 = No, 1 = Yes)
- **Class Distribution:** ~95% No Stroke, ~5% Stroke (20:1 imbalance)
- **Data Split:** 70% Train / 15% Validation / 15% Test

### Features:

#### Demographic (5):
- Gender (Male/Female/Other)
- Age (0-100 years)
- Ever Married (Yes/No)
- Work Type (Never worked/Children/Self-employed/Private/Government)
- Residence Type (Urban/Rural)

#### Clinical (4):
- Hypertension (Yes/No)
- Heart Disease (Yes/No)
- Average Glucose Level (mg/dL)
- BMI (Body Mass Index)

#### Lifestyle (1):
- Smoking Status (Never/Unknown/Formerly/Currently)

---

## 🤖 Machine Learning Models

### Models Evaluated (4):
1. **Logistic Regression** - Interpretable baseline with balanced class weights
2. **Decision Tree** - Rule-based model for feature interaction analysis
3. **Random Forest** - Ensemble method (200 estimators)
4. **MLP (Multi-Layer Perceptron)** - Neural network with 3 hidden layers (100→50→25)

### Comparison Strategy:
**Each model trained in TWO configurations:**
- Without SMOTE-NC (original imbalanced data)
- With SMOTE-NC (balanced synthetic data)

**Total Model Configurations:** 8 (4 models × 2 conditions)

---

## 📈 Evaluation Methodology

### Metrics:
- **F1-Score** - Primary metric (balances precision/recall for imbalanced data)
- **Recall** - Critical for medical diagnosis (find all stroke cases)
- **Precision** - Accuracy of positive predictions
- **ROC-AUC** - Overall discrimination ability

### Validation:
- **Stratified 10-Fold Cross-Validation** - Robust performance estimation
- **Paired t-test** - Statistical significance testing (α = 0.05)
- **Confusion Matrices** - Detailed error analysis

---

## 🔬 Technical Implementation

### SMOTE-NC Configuration:
```python
SMOTENC(
    categorical_features=[0, 2, 3, 4, 5, 6, 9],  # 7 categorical features
    random_state=42,
    k_neighbors=5
)
```
- Balances minority class (stroke cases)
- Handles mixed data types appropriately
- Applied only to training data

### MLP Architecture:
```
Input Layer (10 features)
    ↓
Hidden Layer 1 (100 neurons, ReLU)
    ↓
Hidden Layer 2 (50 neurons, ReLU)
    ↓
Hidden Layer 3 (25 neurons, ReLU)
    ↓
Output Layer (2 classes, Softmax)
```

### Statistical Testing:
- **Null Hypothesis:** No difference between models
- **Test Type:** Paired t-test (dependent samples)
- **Comparisons:** SMOTE vs No-SMOTE, MLP vs other models
- **Significance Levels:** *** p<0.001, ** p<0.01, * p<0.05

---

## 📁 Project Structure

```
├── data/
│   └── data.csv                          # Dataset
├── data_factory/
│   └── data_factory.py                   # Data preprocessing
├── train_models_comprehensive.py         # ⭐ Main training script
├── generate_visualizations.py            # ⭐ Visualization generator
├── app.py                                # Streamlit GUI
├── run_main.py                           # Original training script
├── requirements.txt                      # Dependencies
├── README.md                             # Project overview (original)
├── README_COMPREHENSIVE.md               # ⭐ This file
├── QUICK_START.md                        # ⭐ Quick reference
├── TRAINING_GUIDE.md                     # ⭐ Detailed guide
└── IMPLEMENTATION_SUMMARY.md             # ⭐ Implementation details
```

### Generated Files (After Training):
```
├── best_stroke_model.pkl                 # Best performing model
├── scaler.pkl                            # StandardScaler
├── model_comparison_results.csv          # Summary table
├── kfold_cv_detailed_results.csv         # Detailed CV results
└── figure*.png                           # 6 visualization figures
```

---

## 🎯 Results Overview

### What You'll Get:

#### Console Output:
- Dataset information and class distribution
- SMOTE-NC application results
- Training progress for all 8 model configurations
- K-Fold CV results (Mean ± Std) for each model
- Test set performance metrics
- Comparison table (SMOTE vs No-SMOTE)
- Paired t-test results with significance levels
- Confusion matrices for all models
- Automatic best model selection

#### CSV Files:
- **model_comparison_results.csv** - Summary comparison table
- **kfold_cv_detailed_results.csv** - Fold-by-fold CV results

#### Visualizations (PNG):
1. Model comparison bar charts (4 metrics)
2. K-Fold CV distribution box plots
3. Performance heatmaps (with/without SMOTE)
4. F1-Score across folds line plots
5. SMOTE-NC improvement percentage chart
6. Summary statistics table

---

## 🖥️ GUI Application

### Features:
- **Input Interface:** Form-based patient data entry with validation
- **Real-time Prediction:** Instant stroke risk calculation
- **Risk Visualization:** Color-coded gauge chart (Low/Medium/High)
- **Risk Analysis:** Identification of high-risk factors
- **Recommendations:** Personalized health suggestions
- **Professional Design:** Medical-style interface

### Sample Demo Data:

**High Risk Patient:**
- Age: 75, Hypertension: Yes, Heart Disease: Yes
- Glucose: 200, BMI: 32, Smoking: Yes
- Expected Risk: >60%

**Low Risk Patient:**
- Age: 25, Hypertension: No, Heart Disease: No
- Glucose: 90, BMI: 22, Smoking: No
- Expected Risk: <20%

---

## 📚 Documentation

### Comprehensive Guides Available:

1. **[QUICK_START.md](QUICK_START.md)**
   - 5-minute getting started guide
   - Step-by-step workflow
   - Key concepts explained

2. **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**
   - Detailed training instructions
   - Results interpretation
   - Troubleshooting guide
   - Tips for report and presentation

3. **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**
   - Complete implementation details
   - Technical specifications
   - What's new vs original code

---

## 🎓 Educational Value

### Demonstrates:
- ✅ Medical ML best practices
- ✅ Class imbalance handling (SMOTE-NC)
- ✅ Model validation (K-Fold CV)
- ✅ Statistical rigor (Paired t-test)
- ✅ Deep learning (MLP)
- ✅ Model comparison methodology
- ✅ Reproducible research (fixed seeds)
- ✅ Professional deployment (GUI)
- ✅ Publication-quality visualization

---

## 🔧 Dependencies

```
pandas                # Data manipulation
scikit-learn         # ML models and metrics
numpy                # Numerical computing
matplotlib           # Static plotting
seaborn              # Statistical visualization
scipy                # Statistical tests
streamlit            # Web GUI
imbalanced-learn     # SMOTE-NC
plotly               # Interactive charts
joblib               # Model serialization
```

Install all: `pip install -r requirements.txt`

---

## 📊 Expected Performance

### Typical Results (Approximate):

| Metric | Logistic Reg. | Decision Tree | Random Forest | MLP |
|--------|---------------|---------------|---------------|-----|
| F1-Score (No SMOTE) | ~0.28 | ~0.15 | ~0.27 | ~0.28 |
| F1-Score (SMOTE-NC) | ~0.34 | ~0.21 | ~0.32 | ~0.34 |
| Improvement | +21% | +40% | +18% | +21% |

*Note: Actual results vary - run training to see exact numbers*

### Key Insights:
- SMOTE-NC typically improves recall significantly
- MLP and Logistic Regression often perform best
- Statistical tests confirm improvements are significant
- Class imbalance remains challenging despite mitigation

---

## 🚧 Limitations and Future Work

### Current Limitations:
- Limited to 10 features from dataset
- Class imbalance remains challenging
- Requires external validation
- No temporal analysis

### Future Enhancements:
- Ensemble methods (stacking/blending)
- Feature engineering (interactions)
- Explainable AI (SHAP values)
- Real-time data integration
- Multi-center validation
- Longitudinal analysis

---

## ⚠️ Important Notes

### Medical Disclaimer:
This is an **educational project** for academic purposes only. The predictions should **NOT** be used for actual clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

### Data Privacy:
Dataset is from publicly available Kaggle source. No patient identification information included.

### Reproducibility:
- All random seeds fixed (seed=42)
- Complete code and documentation provided
- Step-by-step instructions available

---

## 📞 Support

### Getting Help:
1. **Check Documentation:** Review QUICK_START.md or TRAINING_GUIDE.md
2. **Verify Installation:** Ensure all dependencies installed
3. **Check Console Output:** Look for error messages
4. **Review Generated Files:** Ensure CSV files created

### Common Issues:
- **"Model files not found"** → Run training script first
- **Package errors** → Update: `pip install --upgrade -r requirements.txt`
- **SMOTE-NC fails** → Check categorical feature indices
- **Slow training** → Normal for neural networks, wait 2-5 min

---

## 📖 References

### Methodology:
1. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique"
2. Kohavi, "A Study of Cross-Validation and Bootstrap for Accuracy Estimation"
3. Student's t-test for dependent samples

### Dataset:
- Stroke Prediction Dataset, Kaggle
- Fedesoriano (contributor)

### Tools:
- Scikit-learn: Pedregosa et al., "Scikit-learn: Machine Learning in Python"
- Imbalanced-learn: Lemaître et al., "Imbalanced-learn: A Python Toolbox"
- Streamlit: Open-source app framework

---

## 🏆 Project Status

- ✅ Data preprocessing complete
- ✅ All models implemented (Logistic, Tree, RF, MLP)
- ✅ SMOTE-NC integration complete
- ✅ K-Fold CV implemented (10 folds)
- ✅ Paired t-test analysis complete
- ✅ GUI development complete
- ✅ Visualization scripts complete
- ✅ Comprehensive documentation complete
- ⏳ Final report in progress
- ⏳ Presentation preparation in progress

---

## 🎉 Acknowledgments

- **Course Instructor** - For valuable feedback on MLP, SMOTE-NC, K-Fold CV, and paired t-test
- **Kaggle Community** - For providing the dataset
- **Open Source Contributors** - Scikit-learn, Streamlit, and related libraries
- **Our Team** - For collaborative effort and dedication

---

## 📄 License

This project is for academic and educational purposes.

---

## 💡 Tips for Final Report

### Include:
1. **Introduction**
   - Problem statement
   - Class imbalance challenge
   - SMOTE-NC motivation

2. **Methods**
   - Dataset description
   - SMOTE-NC technique
   - MLP architecture
   - K-Fold CV methodology
   - Statistical testing approach

3. **Results**
   - Comparison table (all 8 models)
   - K-Fold CV results with std
   - Paired t-test outcomes
   - Best model identification
   - All 6 visualization figures

4. **Discussion**
   - SMOTE-NC impact analysis
   - MLP vs traditional models
   - Statistical significance interpretation
   - Clinical implications
   - Limitations

5. **Conclusion**
   - Best model recommendation
   - Key findings summary
   - Future work directions

---

## 🎤 Presentation Tips

### Suggested Flow:
1. **Problem & Dataset** (2 min)
   - Stroke prediction importance
   - Dataset overview
   - Class imbalance challenge

2. **Methodology** (3 min)
   - SMOTE-NC explanation
   - Models compared (4 models × 2 conditions)
   - K-Fold CV and paired t-test

3. **Results** (4 min)
   - Show comparison table
   - Present key figures
   - Discuss statistical significance
   - Highlight best model

4. **Live Demo** (2 min)
   - Launch Streamlit app
   - Input sample patient
   - Show prediction and explanation

5. **Conclusion** (1 min)
   - Key findings
   - Practical implications
   - Future work

---

**Ready for final submission! 🎓🚀**

For detailed instructions, see:
- [QUICK_START.md](QUICK_START.md) - Quick reference
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Comprehensive guide
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details
