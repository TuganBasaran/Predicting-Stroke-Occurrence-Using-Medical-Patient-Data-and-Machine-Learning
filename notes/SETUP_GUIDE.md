# Stroke Prediction Project - Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Open and run `stroke_prediction_full.ipynb` in Jupyter or VS Code:
- This will train all models (Logistic Regression, Decision Tree, Random Forest, MLP)
- Perform K-Fold Cross-Validation
- Run paired statistical tests
- Save the best model as `best_stroke_model.pkl`

### 3. Run the Web Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 What's Included

### Files:
- `stroke_prediction_full.ipynb` - Complete ML pipeline with K-Fold CV and statistical tests
- `app.py` - Streamlit web application for predictions
- `data_factory/data_factory.py` - Data preprocessing class
- `data/data.csv` - Stroke prediction dataset
- `requirements.txt` - Python dependencies

### Features:
✅ K-Fold Cross-Validation (5 folds)
✅ Paired T-Tests for model comparison  
✅ 4 ML algorithms (Logistic Regression, Decision Tree, Random Forest, MLP)
✅ SMOTE for handling class imbalance
✅ Feature importance analysis
✅ Interactive web GUI with risk visualization
✅ Comprehensive evaluation metrics

## 🎯 Project Structure

```
├── data/
│   └── data.csv                    # Dataset
├── data_factory/
│   └── data_factory.py             # Data preprocessing
├── stroke_prediction_full.ipynb    # Main notebook
├── app.py                          # Streamlit GUI
├── requirements.txt                # Dependencies
├── best_stroke_model.pkl          # Saved model (after training)
├── scaler.pkl                     # Saved scaler (after training)
└── SETUP_GUIDE.md                 # This file
```

## 📝 Usage

### Training:
1. Open `stroke_prediction_full.ipynb`
2. Run all cells sequentially
3. Models will be trained and evaluated
4. Best model will be saved automatically

### Web App:
1. Make sure you've trained the model first
2. Run `streamlit run app.py`
3. Enter patient information in the sidebar
4. Click "Predict Stroke Risk"
5. View risk assessment and recommendations

## 🧪 Models

1. **Logistic Regression** - Linear baseline model
2. **Decision Tree** - Rule-based interpretable model
3. **Random Forest** - Ensemble of decision trees
4. **MLP (Neural Network)** - Deep learning model

## 📈 Evaluation

All models are evaluated using:
- **K-Fold Cross-Validation** (5 folds)
- **Paired T-Tests** for statistical significance
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations**: Confusion matrices, ROC curves, feature importance

## 👥 Authors

- Tugan Başaran
- Zehra Sağın
- Mert Korkmaz

## 📅 Date

January 2026
