from data_factory.data_factory import StrokeDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTENC
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
CATEGORICAL_FEATURES = [0, 2, 3, 4, 5, 6, 9]

print("\nStroke Prediction - Model Comparison")
print(f"K-Fold CV: {K_FOLDS} | Seed: {SEED}\n")

# Load and prepare dataset
dataset = StrokeDataset(path_file, SEED)

print("Dataset Info:")
print(f"  Train: {len(dataset.X_train)} | Val: {len(dataset.X_val)} | Test: {len(dataset.X_test)}")
print(f"  No Stroke: {np.sum(dataset.Y_train == 0)} | Stroke: {np.sum(dataset.Y_train == 1)}")
print(f"  Imbalance ratio: {np.sum(dataset.Y_train == 0) / np.sum(dataset.Y_train == 1):.1f}:1\n")

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(dataset.X_train)
X_val_scaled = scaler.transform(dataset.X_val)
X_test_scaled = scaler.transform(dataset.X_test)

# Apply SMOTE-NC with reduced sampling
print("Applying SMOTE-NC...")
smote_nc = SMOTENC(categorical_features=CATEGORICAL_FEATURES, random_state=SEED, 
                   k_neighbors=5, sampling_strategy=0.5)  # Only balance to 50% instead of 100%
X_train_smote, Y_train_smote = smote_nc.fit_resample(dataset.X_train, dataset.Y_train)
X_train_smote_scaled = scaler.fit_transform(X_train_smote)
print(f"  After SMOTE: {len(X_train_smote)} samples (No Stroke: {np.sum(Y_train_smote == 0)} | Stroke: {np.sum(Y_train_smote == 1)})\n")

# Define models with strong regularization
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, 
        class_weight='balanced', 
        C=0.01,  # Very strong regularization
        penalty='l2',
        solver='lbfgs',
        random_state=SEED
    ),
    'Decision Tree': DecisionTreeClassifier(
        criterion='entropy', 
        max_depth=4,  # Further reduced
        min_samples_split=100,  # More samples required
        min_samples_leaf=50,  # More samples in leaf
        min_impurity_decrease=0.01,  # Minimum improvement to split
        class_weight='balanced',
        random_state=SEED
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=50,  # Fewer trees
        max_depth=6,  # Shallower trees
        min_samples_split=100,
        min_samples_leaf=50,
        max_features='sqrt',
        min_impurity_decrease=0.005,
        class_weight='balanced', 
        random_state=SEED
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(30,),  # Single smaller layer
        activation='relu', 
        solver='adam',
        alpha=0.1,  # Stronger L2 regularization
        max_iter=200,
        early_stopping=True, 
        validation_fraction=0.3,  # More validation data
        n_iter_no_change=20,
        random_state=SEED
    )
}

def perform_k_fold_cv(model, X, y, cv=10):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    return {
        'f1': cross_val_score(model, X, y, cv=skf, scoring='f1', n_jobs=-1),
        'recall': cross_val_score(model, X, y, cv=skf, scoring='recall', n_jobs=-1),
        'precision': cross_val_score(model, X, y, cv=skf, scoring='precision', n_jobs=-1),
        'roc_auc': cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    }

results_without_smote = {}
results_with_smote = {}

# Train without SMOTE-NC
print("Training models WITHOUT SMOTE-NC:\n")
for model_name, model in models.items():
    print(f"  {model_name}...", end=" ")
    cv_scores = perform_k_fold_cv(model, X_train_scaled, dataset.Y_train, cv=K_FOLDS)
    model.fit(X_train_scaled, dataset.Y_train)
    y_pred_test = model.predict(X_test_scaled)
    y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    
    results_without_smote[model_name] = {
        'cv_scores': cv_scores,
        'test_f1': f1_score(dataset.Y_test, y_pred_test),
        'test_recall': recall_score(dataset.Y_test, y_pred_test),
        'test_precision': precision_score(dataset.Y_test, y_pred_test, zero_division=0),
        'test_roc_auc': roc_auc_score(dataset.Y_test, y_proba_test),
        'predictions': y_pred_test,
        'probabilities': y_proba_test,
        'model': model
    }
    print(f"CV F1: {cv_scores['f1'].mean():.3f}±{cv_scores['f1'].std():.3f} | Test F1: {results_without_smote[model_name]['test_f1']:.3f}")

# Train with SMOTE-NC
print("\nTraining models WITH SMOTE-NC:\n")
for model_name in models.keys():
    print(f"  {model_name}...", end=" ")
    
    if model_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000, C=0.01, penalty='l2', solver='lbfgs', random_state=SEED)
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=100, 
                                      min_samples_leaf=50, min_impurity_decrease=0.01,
                                      class_weight='balanced', random_state=SEED)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_split=100,
                                      min_samples_leaf=50, max_features='sqrt', 
                                      min_impurity_decrease=0.005, random_state=SEED)
    elif model_name == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=(30,), activation='relu', solver='adam',
                            alpha=0.1, max_iter=200, early_stopping=True, validation_fraction=0.3,
                            n_iter_no_change=20, random_state=SEED)
    
    cv_scores = perform_k_fold_cv(model, X_train_smote_scaled, Y_train_smote, cv=K_FOLDS)
    model.fit(X_train_smote_scaled, Y_train_smote)
    y_pred_test = model.predict(X_test_scaled)
    y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    
    results_with_smote[model_name] = {
        'cv_scores': cv_scores,
        'test_f1': f1_score(dataset.Y_test, y_pred_test),
        'test_recall': recall_score(dataset.Y_test, y_pred_test),
        'test_precision': precision_score(dataset.Y_test, y_pred_test, zero_division=0),
        'test_roc_auc': roc_auc_score(dataset.Y_test, y_proba_test),
        'predictions': y_pred_test,
        'probabilities': y_proba_test,
        'model': model
    }
    print(f"CV F1: {cv_scores['f1'].mean():.3f}±{cv_scores['f1'].std():.3f} | Test F1: {results_with_smote[model_name]['test_f1']:.3f}")

# Comparison table
print("\n\nComparison Table (Test Set):\n")
comparison_data = []
for model_name in models.keys():
    comparison_data.append({
        'Model': model_name,
        'Condition': 'No SMOTE',
        'F1': f"{results_without_smote[model_name]['test_f1']:.4f}",
        'Recall': f"{results_without_smote[model_name]['test_recall']:.4f}",
        'Precision': f"{results_without_smote[model_name]['test_precision']:.4f}",
        'ROC-AUC': f"{results_without_smote[model_name]['test_roc_auc']:.4f}"
    })
    comparison_data.append({
        'Model': model_name,
        'Condition': 'With SMOTE',
        'F1': f"{results_with_smote[model_name]['test_f1']:.4f}",
        'Recall': f"{results_with_smote[model_name]['test_recall']:.4f}",
        'Precision': f"{results_with_smote[model_name]['test_precision']:.4f}",
        'ROC-AUC': f"{results_with_smote[model_name]['test_roc_auc']:.4f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Paired t-test
print("\n\nPaired t-test (SMOTE vs No SMOTE):\n")
for model_name in models.keys():
    f1_without = results_without_smote[model_name]['cv_scores']['f1']
    f1_with = results_with_smote[model_name]['cv_scores']['f1']
    t_stat, p_value = stats.ttest_rel(f1_with, f1_without)
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    mean_diff = f1_with.mean() - f1_without.mean()
    
    print(f"{model_name}:")
    print(f"  No SMOTE: {f1_without.mean():.4f}±{f1_without.std():.4f} | SMOTE: {f1_with.mean():.4f}±{f1_with.std():.4f}")
    print(f"  Diff: {mean_diff:+.4f} | p={p_value:.4f} {sig}")
    if p_value < 0.05:
        print(f"  → SMOTE is {'better' if mean_diff > 0 else 'worse'} (p<0.05)")
    print()

# MLP comparison
print("MLP vs Other Models (with SMOTE):\n")
mlp_scores = results_with_smote['MLP']['cv_scores']['f1']
for model_name in models.keys():
    if model_name != 'MLP':
        other_scores = results_with_smote[model_name]['cv_scores']['f1']
        t_stat, p_value = stats.ttest_rel(mlp_scores, other_scores)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        mean_diff = mlp_scores.mean() - other_scores.mean()
        
        print(f"MLP vs {model_name}: Diff={mean_diff:+.4f} | p={p_value:.4f} {sig}")
        if p_value < 0.05:
            winner = "MLP" if mean_diff > 0 else model_name
            print(f"  → {winner} is better")

# Confusion matrices
print("\n\nConfusion Matrices (Test Set):\n")
for model_name in models.keys():
    print(f"{model_name}:")
    print(f"  No SMOTE: {confusion_matrix(dataset.Y_test, results_without_smote[model_name]['predictions']).ravel()}")
    print(f"  SMOTE:    {confusion_matrix(dataset.Y_test, results_with_smote[model_name]['predictions']).ravel()}")

# Select best model
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

print(f"\n\nBest Model: {best_model_name} {'WITH SMOTE' if best_with_smote else 'WITHOUT SMOTE'}")
print(f"  Test F1: {best_f1_score:.4f}")

best_model = results_with_smote[best_model_name]['model'] if best_with_smote else results_without_smote[best_model_name]['model']
stats_dict = results_with_smote[best_model_name] if best_with_smote else results_without_smote[best_model_name]
print(f"  Recall: {stats_dict['test_recall']:.4f} | Precision: {stats_dict['test_precision']:.4f} | ROC-AUC: {stats_dict['test_roc_auc']:.4f}")

# Save results
joblib.dump(best_model, 'best_stroke_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
comparison_df.to_csv('model_comparison_results.csv', index=False)

cv_results_data = []
for model_name in models.keys():
    for i in range(K_FOLDS):
        cv_results_data.append({
            'Model': model_name, 'Condition': 'Without SMOTE-NC', 'Fold': i+1,
            'F1': results_without_smote[model_name]['cv_scores']['f1'][i],
            'Recall': results_without_smote[model_name]['cv_scores']['recall'][i],
            'Precision': results_without_smote[model_name]['cv_scores']['precision'][i],
            'ROC_AUC': results_without_smote[model_name]['cv_scores']['roc_auc'][i]
        })
        cv_results_data.append({
            'Model': model_name, 'Condition': 'With SMOTE-NC', 'Fold': i+1,
            'F1': results_with_smote[model_name]['cv_scores']['f1'][i],
            'Recall': results_with_smote[model_name]['cv_scores']['recall'][i],
            'Precision': results_with_smote[model_name]['cv_scores']['precision'][i],
            'ROC_AUC': results_with_smote[model_name]['cv_scores']['roc_auc'][i]
        })

pd.DataFrame(cv_results_data).to_csv('kfold_cv_detailed_results.csv', index=False)

print("\n\nSaved:")
print("  - best_stroke_model.pkl")
print("  - scaler.pkl")
print("  - model_comparison_results.csv")
print("  - kfold_cv_detailed_results.csv")
print("\nDone! Run 'streamlit run app.py' for GUI")
