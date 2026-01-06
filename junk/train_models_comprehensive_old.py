from data_factory.data_factory import StrokeDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
path_file = "data/data.csv"
SEED = 42
K_FOLDS = 10

# Categorical feature indices
CATEGORICAL_FEATURES = [0, 2, 3, 4, 5, 6, 9]

print("\nSTROKE PREDICTION - MODEL COMPARISON")
print(f"K-Fold CV: {K_FOLDS} | Random Seed: {SEED}\n")

# Load dataset
dataset = StrokeDataset(path_file, SEED)

print("\n📊 Dataset Information:")
print(f"Training samples: {len(dataset.X_train)}")
print(f"Validation samples: {len(dataset.X_val)}")
print(f"Test samples: {len(dataset.X_test)}")
print(f"Features: {dataset.X_features}")
print(f"\nClass distribution in training set:")
print(f"  No Stroke (0): {np.sum(dataset.Y_train == 0)}")
print(f"  Stroke (1): {np.sum(dataset.Y_train == 1)}")
print(f"  Imbalance ratio: {np.sum(dataset.Y_train == 0) / np.sum(dataset.Y_train == 1):.2f}:1")

# =====================================
# DATA SCALING
# =====================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(dataset.X_train)
X_val_scaled = scaler.transform(dataset.X_val)
X_test_scaled = scaler.transform(dataset.X_test)

# =====================================
# APPLY SMOTE-NC FOR OVERSAMPLING
# =====================================

print("\n" + "=" * 80)
print("APPLYING SMOTE-NC (Synthetic Minority Over-sampling)")
print("=" * 80)

smote_nc = SMOTENC(categorical_features=CATEGORICAL_FEATURES, random_state=SEED, k_neighbors=5)
X_train_smote, Y_train_smote = smote_nc.fit_resample(dataset.X_train, dataset.Y_train)

print(f"\n📈 SMOTE-NC Results:")
print(f"Original training samples: {len(dataset.X_train)}")
print(f"After SMOTE-NC: {len(X_train_smote)}")
print(f"  No Stroke (0): {np.sum(Y_train_smote == 0)}")
print(f"  Stroke (1): {np.sum(Y_train_smote == 1)}")
print(f"  New ratio: {np.sum(Y_train_smote == 0) / np.sum(Y_train_smote == 1):.2f}:1")

# Scale SMOTE data
X_train_smote_scaled = scaler.fit_transform(X_train_smote)

# =====================================
# DEFINE MODELS
# =====================================

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED),
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=None, class_weight='balanced', random_state=SEED),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', 
                         max_iter=500, random_state=SEED, early_stopping=True, validation_fraction=0.15)
}

# =====================================
# K-FOLD CROSS-VALIDATION FUNCTION
# =====================================

def perform_k_fold_cv(model, X, y, cv=10):
    """
    Perform K-Fold Cross-Validation and return scores
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    
    # Different scoring metrics
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1', n_jobs=-1)
    recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall', n_jobs=-1)
    precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision', n_jobs=-1)
    roc_auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    
    return {
        'f1': f1_scores,
        'recall': recall_scores,
        'precision': precision_scores,
        'roc_auc': roc_auc_scores
    }

# =====================================
# TRAIN AND EVALUATE MODELS
# =====================================

results_without_smote = {}
results_with_smote = {}
test_predictions = {}

print("\n" + "=" * 80)
print("TRAINING MODELS WITHOUT SMOTE-NC")
print("=" * 80)

for model_name, model in models.items():
    print(f"\n🔄 Training {model_name}...")
    
    # K-Fold Cross-Validation
    cv_scores = perform_k_fold_cv(model, X_train_scaled, dataset.Y_train, cv=K_FOLDS)
    
    # Train on full training set
    model.fit(X_train_scaled, dataset.Y_train)
    
    # Predict on test set
    y_pred_test = model.predict(X_test_scaled)
    y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate test metrics
    test_f1 = f1_score(dataset.Y_test, y_pred_test)
    test_recall = recall_score(dataset.Y_test, y_pred_test)
    test_precision = precision_score(dataset.Y_test, y_pred_test, zero_division=0)
    test_roc_auc = roc_auc_score(dataset.Y_test, y_proba_test)
    
    results_without_smote[model_name] = {
        'cv_scores': cv_scores,
        'test_f1': test_f1,
        'test_recall': test_recall,
        'test_precision': test_precision,
        'test_roc_auc': test_roc_auc,
        'predictions': y_pred_test,
        'probabilities': y_proba_test,
        'model': model
    }
    
    # Store predictions for paired t-test
    test_predictions[f'{model_name}_no_smote'] = y_pred_test
    
    print(f"  ✓ K-Fold CV Results (Mean ± Std):")
    print(f"    F1-Score:    {cv_scores['f1'].mean():.4f} ± {cv_scores['f1'].std():.4f}")
    print(f"    Recall:      {cv_scores['recall'].mean():.4f} ± {cv_scores['recall'].std():.4f}")
    print(f"    Precision:   {cv_scores['precision'].mean():.4f} ± {cv_scores['precision'].std():.4f}")
    print(f"    ROC-AUC:     {cv_scores['roc_auc'].mean():.4f} ± {cv_scores['roc_auc'].std():.4f}")
    print(f"  ✓ Test Set Results:")
    print(f"    F1-Score:    {test_f1:.4f}")
    print(f"    Recall:      {test_recall:.4f}")
    print(f"    Precision:   {test_precision:.4f}")
    print(f"    ROC-AUC:     {test_roc_auc:.4f}")

print("\n" + "=" * 80)
print("TRAINING MODELS WITH SMOTE-NC")
print("=" * 80)

for model_name, model_class in models.items():
    print(f"\n🔄 Training {model_name} with SMOTE-NC...")
    
    # Create new model instance
    if model_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000, random_state=SEED)
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=SEED)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=SEED)
    elif model_name == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam',
                            max_iter=500, random_state=SEED, early_stopping=True, validation_fraction=0.15)
    
    # K-Fold Cross-Validation with SMOTE
    cv_scores = perform_k_fold_cv(model, X_train_smote_scaled, Y_train_smote, cv=K_FOLDS)
    
    # Train on full SMOTE training set
    model.fit(X_train_smote_scaled, Y_train_smote)
    
    # Predict on test set
    y_pred_test = model.predict(X_test_scaled)
    y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate test metrics
    test_f1 = f1_score(dataset.Y_test, y_pred_test)
    test_recall = recall_score(dataset.Y_test, y_pred_test)
    test_precision = precision_score(dataset.Y_test, y_pred_test, zero_division=0)
    test_roc_auc = roc_auc_score(dataset.Y_test, y_proba_test)
    
    results_with_smote[model_name] = {
        'cv_scores': cv_scores,
        'test_f1': test_f1,
        'test_recall': test_recall,
        'test_precision': test_precision,
        'test_roc_auc': test_roc_auc,
        'predictions': y_pred_test,
        'probabilities': y_proba_test,
        'model': model
    }
    
    # Store predictions for paired t-test
    test_predictions[f'{model_name}_smote'] = y_pred_test
    
    print(f"  ✓ K-Fold CV Results (Mean ± Std):")
    print(f"    F1-Score:    {cv_scores['f1'].mean():.4f} ± {cv_scores['f1'].std():.4f}")
    print(f"    Recall:      {cv_scores['recall'].mean():.4f} ± {cv_scores['recall'].std():.4f}")
    print(f"    Precision:   {cv_scores['precision'].mean():.4f} ± {cv_scores['precision'].std():.4f}")
    print(f"    ROC-AUC:     {cv_scores['roc_auc'].mean():.4f} ± {cv_scores['roc_auc'].std():.4f}")
    print(f"  ✓ Test Set Results:")
    print(f"    F1-Score:    {test_f1:.4f}")
    print(f"    Recall:      {test_recall:.4f}")
    print(f"    Precision:   {test_precision:.4f}")
    print(f"    ROC-AUC:     {test_roc_auc:.4f}")

# =====================================
# COMPARISON TABLE
# =====================================

print("\n" + "=" * 80)
print("COMPARISON: SMOTE-NC vs NO SMOTE-NC (Test Set Performance)")
print("=" * 80)

comparison_data = []
for model_name in models.keys():
    comparison_data.append({
        'Model': model_name,
        'Condition': 'Without SMOTE-NC',
        'F1-Score': f"{results_without_smote[model_name]['test_f1']:.4f}",
        'Recall': f"{results_without_smote[model_name]['test_recall']:.4f}",
        'Precision': f"{results_without_smote[model_name]['test_precision']:.4f}",
        'ROC-AUC': f"{results_without_smote[model_name]['test_roc_auc']:.4f}"
    })
    comparison_data.append({
        'Model': model_name,
        'Condition': 'With SMOTE-NC',
        'F1-Score': f"{results_with_smote[model_name]['test_f1']:.4f}",
        'Recall': f"{results_with_smote[model_name]['test_recall']:.4f}",
        'Precision': f"{results_with_smote[model_name]['test_precision']:.4f}",
        'ROC-AUC': f"{results_with_smote[model_name]['test_roc_auc']:.4f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# =====================================
# PAIRED T-TEST ANALYSIS
# =====================================

print("\n" + "=" * 80)
print("PAIRED T-TEST: Statistical Comparison")
print("=" * 80)
print("\nComparing models WITH and WITHOUT SMOTE-NC using K-Fold CV F1-Scores\n")

for model_name in models.keys():
    f1_without = results_without_smote[model_name]['cv_scores']['f1']
    f1_with = results_with_smote[model_name]['cv_scores']['f1']
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(f1_with, f1_without)
    
    # Determine significance
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    
    mean_diff = f1_with.mean() - f1_without.mean()
    
    print(f"🔬 {model_name}:")
    print(f"   Mean F1 (No SMOTE):   {f1_without.mean():.4f} ± {f1_without.std():.4f}")
    print(f"   Mean F1 (SMOTE-NC):   {f1_with.mean():.4f} ± {f1_with.std():.4f}")
    print(f"   Difference:           {mean_diff:+.4f}")
    print(f"   t-statistic:          {t_stat:.4f}")
    print(f"   p-value:              {p_value:.4f} {significance}")
    
    if p_value < 0.05:
        direction = "better" if mean_diff > 0 else "worse"
        print(f"   ✓ SMOTE-NC is significantly {direction} (p < 0.05)")
    else:
        print(f"   → No significant difference detected")
    print()

print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")

# =====================================
# CROSS-MODEL COMPARISON (Paired t-test between different models)
# =====================================

print("\n" + "=" * 80)
print("CROSS-MODEL COMPARISON (Paired T-Test)")
print("=" * 80)
print("\nComparing MLP against other models (with SMOTE-NC)\n")

model_names_list = list(models.keys())
mlp_scores = results_with_smote['MLP']['cv_scores']['f1']

for model_name in model_names_list:
    if model_name != 'MLP':
        other_scores = results_with_smote[model_name]['cv_scores']['f1']
        t_stat, p_value = stats.ttest_rel(mlp_scores, other_scores)
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        mean_diff = mlp_scores.mean() - other_scores.mean()
        
        print(f"🔬 MLP vs {model_name}:")
        print(f"   MLP F1:               {mlp_scores.mean():.4f} ± {mlp_scores.std():.4f}")
        print(f"   {model_name} F1:      {other_scores.mean():.4f} ± {other_scores.std():.4f}")
        print(f"   Difference:           {mean_diff:+.4f}")
        print(f"   p-value:              {p_value:.4f} {significance}")
        
        if p_value < 0.05:
            winner = "MLP" if mean_diff > 0 else model_name
            print(f"   ✓ {winner} performs significantly better")
        else:
            print(f"   → No significant difference")
        print()

# =====================================
# CONFUSION MATRICES
# =====================================

print("\n" + "=" * 80)
print("CONFUSION MATRICES (Test Set)")
print("=" * 80)

for model_name in models.keys():
    print(f"\n📊 {model_name}:")
    
    print("\n  Without SMOTE-NC:")
    cm_no_smote = confusion_matrix(dataset.Y_test, results_without_smote[model_name]['predictions'])
    print(f"    {cm_no_smote}")
    
    print("\n  With SMOTE-NC:")
    cm_smote = confusion_matrix(dataset.Y_test, results_with_smote[model_name]['predictions'])
    print(f"    {cm_smote}")

# =====================================
# FIND BEST MODEL
# =====================================

print("\n" + "=" * 80)
print("BEST MODEL SELECTION")
print("=" * 80)

# Based on F1-Score on test set
best_model_name = None
best_f1_score = 0
best_with_smote = False

for model_name in models.keys():
    if results_without_smote[model_name]['test_f1'] > best_f1_score:
        best_f1_score = results_without_smote[model_name]['test_f1']
        best_model_name = model_name
        best_with_smote = False
    
    if results_with_smote[model_name]['test_f1'] > best_f1_score:
        best_f1_score = results_with_smote[model_name]['test_f1']
        best_model_name = model_name
        best_with_smote = True

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Configuration: {'WITH SMOTE-NC' if best_with_smote else 'WITHOUT SMOTE-NC'}")
print(f"   Test F1-Score: {best_f1_score:.4f}")

if best_with_smote:
    best_model = results_with_smote[best_model_name]['model']
    print(f"   Test Recall: {results_with_smote[best_model_name]['test_recall']:.4f}")
    print(f"   Test Precision: {results_with_smote[best_model_name]['test_precision']:.4f}")
    print(f"   Test ROC-AUC: {results_with_smote[best_model_name]['test_roc_auc']:.4f}")
else:
    best_model = results_without_smote[best_model_name]['model']
    print(f"   Test Recall: {results_without_smote[best_model_name]['test_recall']:.4f}")
    print(f"   Test Precision: {results_without_smote[best_model_name]['test_precision']:.4f}")
    print(f"   Test ROC-AUC: {results_without_smote[best_model_name]['test_roc_auc']:.4f}")

# =====================================
# SAVE BEST MODEL
# =====================================

print("\n" + "=" * 80)
print("SAVING BEST MODEL")
print("=" * 80)

joblib.dump(best_model, 'best_stroke_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"\n✓ Best model saved as: best_stroke_model.pkl")
print(f"✓ Scaler saved as: scaler.pkl")

# =====================================
# SAVE DETAILED RESULTS TO CSV
# =====================================

print("\n" + "=" * 80)
print("SAVING RESULTS TO CSV")
print("=" * 80)

# Save comparison table
comparison_df.to_csv('model_comparison_results.csv', index=False)
print("✓ Comparison table saved as: model_comparison_results.csv")

# Save detailed K-Fold CV results
cv_results_data = []
for model_name in models.keys():
    for i in range(K_FOLDS):
        cv_results_data.append({
            'Model': model_name,
            'Condition': 'Without SMOTE-NC',
            'Fold': i+1,
            'F1': results_without_smote[model_name]['cv_scores']['f1'][i],
            'Recall': results_without_smote[model_name]['cv_scores']['recall'][i],
            'Precision': results_without_smote[model_name]['cv_scores']['precision'][i],
            'ROC_AUC': results_without_smote[model_name]['cv_scores']['roc_auc'][i]
        })
        cv_results_data.append({
            'Model': model_name,
            'Condition': 'With SMOTE-NC',
            'Fold': i+1,
            'F1': results_with_smote[model_name]['cv_scores']['f1'][i],
            'Recall': results_with_smote[model_name]['cv_scores']['recall'][i],
            'Precision': results_with_smote[model_name]['cv_scores']['precision'][i],
            'ROC_AUC': results_with_smote[model_name]['cv_scores']['roc_auc'][i]
        })

cv_results_df = pd.DataFrame(cv_results_data)
cv_results_df.to_csv('kfold_cv_detailed_results.csv', index=False)
print("✓ K-Fold CV results saved as: kfold_cv_detailed_results.csv")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print("\n✓ All models trained and evaluated")
print("✓ Statistical tests completed")
print("✓ Best model saved for deployment")
print("\nYou can now run the Streamlit app (app.py) for predictions!")
