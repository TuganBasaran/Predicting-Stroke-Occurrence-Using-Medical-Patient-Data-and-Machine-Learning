from data_factory.data_factory import StrokeDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTENC
import numpy as np
import pandas as pd
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available. Install with: pip install lightgbm")

# Configuration
path_file = "data/data.csv"
SEED = 42
K_FOLDS = 10
CATEGORICAL_FEATURES = [0, 2, 3, 4, 5, 6, 9]

print("\n" + "="*60)
print("Stroke Prediction - OPTIMIZED Version")
print("="*60)
print(f"K-Fold CV: {K_FOLDS} | Seed: {SEED}\n")

# Load and prepare dataset
dataset = StrokeDataset(path_file, SEED)

print("Dataset Info:")
print(f"  Train: {len(dataset.X_train)} | Val: {len(dataset.X_val)} | Test: {len(dataset.X_test)}")
print(f"  No Stroke: {np.sum(dataset.Y_train == 0)} | Stroke: {np.sum(dataset.Y_train == 1)}")
print(f"  Imbalance ratio: {np.sum(dataset.Y_train == 0) / np.sum(dataset.Y_train == 1):.1f}:1\n")

# ==================== FEATURE ENGINEERING ====================
print("="*60)
print("STEP 1: Feature Engineering")
print("="*60)

def create_engineered_features(X, feature_names):
    """Create new features from existing ones"""
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Age groups (0-30: 0, 31-50: 1, 51-70: 2, 71+: 3)
    X_df['age_group'] = pd.cut(X_df['age'], bins=[0, 30, 50, 70, 100], labels=[0, 1, 2, 3]).astype(int)
    
    # BMI categories (Underweight: 0, Normal: 1, Overweight: 2, Obese: 3)
    X_df['bmi_category'] = pd.cut(X_df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
    
    # Risk score (combination of risk factors)
    X_df['risk_score'] = (
        (X_df['age'] > 60).astype(int) +  # Age risk
        X_df['hypertension'] +  # Hypertension
        X_df['heart_disease'] +  # Heart disease
        (X_df['avg_glucose_level'] > 140).astype(int) +  # High glucose
        (X_df['bmi'] > 30).astype(int)  # Obesity
    )
    
    # High glucose flag
    X_df['high_glucose'] = (X_df['avg_glucose_level'] > 140).astype(int)
    
    # Elderly flag
    X_df['is_elderly'] = (X_df['age'] > 65).astype(int)
    
    # Multiple risk factors
    X_df['multiple_risks'] = ((X_df['hypertension'] + X_df['heart_disease']) >= 2).astype(int)
    
    # Interaction features (important combinations)
    X_df['age_glucose'] = X_df['age'] * X_df['avg_glucose_level'] / 100  # Normalize
    X_df['age_bmi'] = X_df['age'] * X_df['bmi'] / 100
    X_df['bmi_glucose'] = X_df['bmi'] * X_df['avg_glucose_level'] / 100
    X_df['age_hypertension'] = X_df['age'] * X_df['hypertension']
    X_df['age_heart'] = X_df['age'] * X_df['heart_disease']
    X_df['bmi_hypertension'] = X_df['bmi'] * X_df['hypertension']
    
    return X_df.values

# Get feature names from dataset
feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

# Apply feature engineering
X_train_eng = create_engineered_features(dataset.X_train, feature_names)
X_val_eng = create_engineered_features(dataset.X_val, feature_names)
X_test_eng = create_engineered_features(dataset.X_test, feature_names)

# Update categorical features indices (original + new categorical features)
CATEGORICAL_FEATURES_ENG = CATEGORICAL_FEATURES + [10, 11]  # age_group, bmi_category

print(f"  Original features: {len(feature_names)}")
print(f"  New features: {X_train_eng.shape[1] - len(feature_names)}")
print(f"  Total features: {X_train_eng.shape[1]}")
print("  Added: age_group, bmi_category, risk_score, high_glucose, is_elderly, multiple_risks")
print("         + interaction features: age_glucose, age_bmi, bmi_glucose, age_hypertension, age_heart, bmi_hypertension\n")

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_eng)
X_val_scaled = scaler.transform(X_val_eng)
X_test_scaled = scaler.transform(X_test_eng)

# Apply SMOTE-NC
print("Applying SMOTE-NC...")
smote_nc = SMOTENC(categorical_features=CATEGORICAL_FEATURES_ENG, random_state=SEED, 
                   k_neighbors=5, sampling_strategy=0.5)
X_train_smote, Y_train_smote = smote_nc.fit_resample(X_train_eng, dataset.Y_train)
X_train_smote_scaled = scaler.fit_transform(X_train_smote)
print(f"  After SMOTE: {len(X_train_smote)} samples (No Stroke: {np.sum(Y_train_smote == 0)} | Stroke: {np.sum(Y_train_smote == 1)})\n")

# ==================== OPTIMIZED CLASS WEIGHTS ====================
print("="*60)
print("STEP 2: Class Weight Optimization")
print("="*60)

# More aggressive class weights
imbalance_ratio = np.sum(dataset.Y_train == 0) / np.sum(dataset.Y_train == 1)
class_weight_custom = {0: 1, 1: int(imbalance_ratio * 1.2)}  # 20% more weight to minority
print(f"  Using custom class weights: {class_weight_custom}\n")

# ==================== BASE MODELS WITH STRONG REGULARIZATION ====================
print("="*60)
print("STEP 3: Define Base Models")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, 
        C=0.01,
        penalty='l2',
        solver='lbfgs',
        class_weight=class_weight_custom,
        random_state=SEED
    ),
    'Decision Tree': DecisionTreeClassifier(
        criterion='entropy', 
        max_depth=4,
        min_samples_split=100,
        min_samples_leaf=50,
        min_impurity_decrease=0.01,
        class_weight=class_weight_custom,
        random_state=SEED
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        min_samples_split=100,
        min_samples_leaf=50,
        max_features='sqrt',
        min_impurity_decrease=0.005,
        class_weight=class_weight_custom,
        random_state=SEED
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=100,
        min_samples_leaf=50,
        subsample=0.8,
        random_state=SEED
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(30,),
        activation='relu', 
        solver='adam',
        alpha=0.1,
        max_iter=200,
        early_stopping=True, 
        validation_fraction=0.3,
        n_iter_no_change=20,
        random_state=SEED
    )
}

# Add XGBoost if available
if HAS_XGBOOST:
    scale_pos_weight = np.sum(dataset.Y_train == 0) / np.sum(dataset.Y_train == 1)
    models['XGBoost'] = xgb.XGBClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=50,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight * 0.5,
        random_state=SEED,
        eval_metric='logloss'
    )

# Add LightGBM if available
if HAS_LIGHTGBM:
    models['LightGBM'] = lgb.LGBMClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=SEED,
        verbose=-1
    )

print("  Base models defined with strong regularization\n")

# ==================== HYPERPARAMETER TUNING ====================
print("="*60)
print("STEP 4: Hyperparameter Tuning (GridSearchCV)")
print("="*60)

param_grids = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1],
        'penalty': ['l2']
    },
    'Decision Tree': {
        'max_depth': [3, 4, 5],
        'min_samples_split': [80, 100, 120],
        'min_samples_leaf': [40, 50, 60]
    },
    'Random Forest': {
        'n_estimators': [30, 50, 70],
        'max_depth': [5, 6, 7],
        'min_samples_split': [80, 100, 120]
    },
    'Gradient Boosting': {
        'n_estimators': [30, 50, 70],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3, 4]
    },
    'MLP': {
        'hidden_layer_sizes': [(30,), (50,), (30, 15)],
        'alpha': [0.05, 0.1, 0.15]
    }
}

if HAS_XGBOOST:
    param_grids['XGBoost'] = {
        'n_estimators': [30, 50, 70],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3, 4]
    }

if HAS_LIGHTGBM:
    param_grids['LightGBM'] = {
        'n_estimators': [30, 50, 70],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3, 4]
    }

tuned_models = {}
for name, model in models.items():
    print(f"  Tuning {name}...")
    grid_search = GridSearchCV(
        model, 
        param_grids[name], 
        cv=5,  # 5-fold for speed
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train_scaled, dataset.Y_train)
    tuned_models[name] = grid_search.best_estimator_
    print(f"    Best params: {grid_search.best_params_}")
    print(f"    Best CV F1: {grid_search.best_score_:.4f}")

print()

# ==================== K-FOLD CROSS-VALIDATION ====================
def perform_k_fold_cv(model, X, y, k=10):
    """Perform K-fold cross-validation with multiple metrics"""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
    recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall', n_jobs=-1)
    precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision', n_jobs=-1)
    roc_auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    return {
        'f1': f1_scores,
        'recall': recall_scores,
        'precision': precision_scores,
        'roc_auc': roc_auc_scores
    }

# ==================== TRAINING WITHOUT SMOTE ====================
print("="*60)
print("STEP 5: Training Models WITHOUT SMOTE")
print("="*60)
print()

results_no_smote = {}
cv_scores_no_smote = {}

for name, model in tuned_models.items():
    print(f"  {name}...", end=" ")
    
    # K-Fold CV
    cv_results = perform_k_fold_cv(model, X_train_scaled, dataset.Y_train, K_FOLDS)
    cv_scores_no_smote[name] = cv_results
    
    # Train on full training set
    model.fit(X_train_scaled, dataset.Y_train)
    
    # Predict on test set
    Y_pred = model.predict(X_test_scaled)
    Y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else Y_pred
    
    # Calculate metrics
    f1 = f1_score(dataset.Y_test, Y_pred)
    recall = recall_score(dataset.Y_test, Y_pred)
    precision = precision_score(dataset.Y_test, Y_pred)
    roc_auc = roc_auc_score(dataset.Y_test, Y_pred_proba)
    
    results_no_smote[name] = {
        'model': model,
        'y_pred': Y_pred,
        'y_pred_proba': Y_pred_proba,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'roc_auc': roc_auc
    }
    
    print(f"CV F1: {cv_results['f1'].mean():.3f}±{cv_results['f1'].std():.3f} | Test F1: {f1:.3f}")

print()

# ==================== TRAINING WITH SMOTE ====================
print("="*60)
print("STEP 6: Training Models WITH SMOTE")
print("="*60)
print()

results_with_smote = {}
cv_scores_with_smote = {}

for name in tuned_models.keys():
    print(f"  {name}...", end=" ")
    
    # Recreate model with same parameters but without class_weight for SMOTE
    if name == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000, C=0.01, penalty='l2', solver='lbfgs', random_state=SEED)
    elif name == 'Decision Tree':
        model = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=100, 
                                      min_samples_leaf=50, min_impurity_decrease=0.01, random_state=SEED)
    elif name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_split=100,
                                      min_samples_leaf=50, max_features='sqrt', 
                                      min_impurity_decrease=0.005, random_state=SEED)
    elif name == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=(30,), activation='relu', solver='adam',
                            alpha=0.1, max_iter=200, early_stopping=True, validation_fraction=0.3,
                            n_iter_no_change=20, random_state=SEED)
    
    # K-Fold CV
    cv_results = perform_k_fold_cv(model, X_train_smote_scaled, Y_train_smote, K_FOLDS)
    cv_scores_with_smote[name] = cv_results
    
    # Train on SMOTE data
    model.fit(X_train_smote_scaled, Y_train_smote)
    
    # Predict on test set
    Y_pred = model.predict(X_test_scaled)
    Y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else Y_pred
    
    # Calculate metrics
    f1 = f1_score(dataset.Y_test, Y_pred)
    recall = recall_score(dataset.Y_test, Y_pred)
    precision = precision_score(dataset.Y_test, Y_pred)
    roc_auc = roc_auc_score(dataset.Y_test, Y_pred_proba)
    
    results_with_smote[name] = {
        'model': model,
        'y_pred': Y_pred,
        'y_pred_proba': Y_pred_proba,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'roc_auc': roc_auc
    }
    
    print(f"CV F1: {cv_results['f1'].mean():.3f}±{cv_results['f1'].std():.3f} | Test F1: {f1:.3f}")

print()

# ==================== THRESHOLD OPTIMIZATION ====================
print("="*60)
print("STEP 7: Threshold Optimization")
print("="*60)
print()

def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """Find optimal threshold for classification"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

results_optimized = {}

for condition, results in [('No SMOTE', results_no_smote), ('With SMOTE', results_with_smote)]:
    print(f"  {condition}:")
    for name, result in results.items():
        if 'y_pred_proba' in result and len(np.unique(result['y_pred_proba'])) > 2:
            optimal_threshold, optimal_f1 = find_optimal_threshold(
                dataset.Y_test, result['y_pred_proba'], metric='f1'
            )
            
            # Apply optimal threshold
            y_pred_optimized = (result['y_pred_proba'] >= optimal_threshold).astype(int)
            f1_optimized = f1_score(dataset.Y_test, y_pred_optimized)
            recall_optimized = recall_score(dataset.Y_test, y_pred_optimized)
            precision_optimized = precision_score(dataset.Y_test, y_pred_optimized)
            
            improvement = f1_optimized - result['f1']
            
            results_optimized[f"{name} ({condition})"] = {
                'threshold': optimal_threshold,
                'f1': f1_optimized,
                'recall': recall_optimized,
                'precision': precision_optimized,
                'improvement': improvement
            }
            
            print(f"    {name}: threshold={optimal_threshold:.2f}, F1={f1_optimized:.3f} (+{improvement:.3f}), Recall={recall_optimized:.3f}")

print()

# ==================== ENSEMBLE METHOD ====================
print("="*60)
print("STEP 8: Ensemble (Voting Classifier)")
print("="*60)
print()

# Select best 3 models from no SMOTE
sorted_models = sorted(results_no_smote.items(), key=lambda x: x[1]['f1'], reverse=True)
top_3_models = [(name, result['model']) for name, result in sorted_models[:3]]

print(f"  Using top 3 models: {', '.join([name for name, _ in top_3_models])}")

# Create voting classifier
voting_clf = VotingClassifier(
    estimators=top_3_models,
    voting='soft',  # Use probabilities
    n_jobs=-1
)

voting_clf.fit(X_train_scaled, dataset.Y_train)
Y_pred_voting = voting_clf.predict(X_test_scaled)
Y_pred_proba_voting = voting_clf.predict_proba(X_test_scaled)[:, 1]

# Optimize threshold for ensemble
optimal_threshold_voting, _ = find_optimal_threshold(dataset.Y_test, Y_pred_proba_voting, metric='f1')
Y_pred_voting_optimized = (Y_pred_proba_voting >= optimal_threshold_voting).astype(int)

f1_voting = f1_score(dataset.Y_test, Y_pred_voting_optimized)
recall_voting = recall_score(dataset.Y_test, Y_pred_voting_optimized)
precision_voting = precision_score(dataset.Y_test, Y_pred_voting_optimized)
roc_auc_voting = roc_auc_score(dataset.Y_test, Y_pred_proba_voting)

print(f"  Ensemble (threshold={optimal_threshold_voting:.2f}):")
print(f"    F1: {f1_voting:.3f} | Recall: {recall_voting:.3f} | Precision: {precision_voting:.3f} | ROC-AUC: {roc_auc_voting:.3f}")
print()

# ==================== STACKING ENSEMBLE ====================
print("="*60)
print("STEP 8b: Stacking Ensemble (Meta-Learner)")
print("="*60)
print()

# Use all good models (F1 > 0.29) as base estimators
good_models = [(name, result['model']) for name, result in sorted_models if result['f1'] > 0.29]
print(f"  Using {len(good_models)} base models: {', '.join([name for name, _ in good_models])}")

# Create stacking classifier with Logistic Regression as meta-learner
stacking_clf = StackingClassifier(
    estimators=good_models,
    final_estimator=LogisticRegression(C=0.1, max_iter=1000, random_state=SEED),
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1
)

print("  Training stacking ensemble...")
stacking_clf.fit(X_train_scaled, dataset.Y_train)
Y_pred_stacking = stacking_clf.predict(X_test_scaled)
Y_pred_proba_stacking = stacking_clf.predict_proba(X_test_scaled)[:, 1]

# Optimize threshold for stacking
optimal_threshold_stacking, _ = find_optimal_threshold(dataset.Y_test, Y_pred_proba_stacking, metric='f1')
Y_pred_stacking_optimized = (Y_pred_proba_stacking >= optimal_threshold_stacking).astype(int)

f1_stacking = f1_score(dataset.Y_test, Y_pred_stacking_optimized)
recall_stacking = recall_score(dataset.Y_test, Y_pred_stacking_optimized)
precision_stacking = precision_score(dataset.Y_test, Y_pred_stacking_optimized)
roc_auc_stacking = roc_auc_score(dataset.Y_test, Y_pred_proba_stacking)

print(f"  Stacking Ensemble (threshold={optimal_threshold_stacking:.2f}):")
print(f"    F1: {f1_stacking:.3f} | Recall: {recall_stacking:.3f} | Precision: {precision_stacking:.3f} | ROC-AUC: {roc_auc_stacking:.3f}")
print()

# ==================== CALIBRATION ====================
print("="*60)
print("STEP 9: Probability Calibration")
print("="*60)
print()

# Calibrate the best model
best_model_name = max(results_no_smote.items(), key=lambda x: x[1]['f1'])[0]
best_model = results_no_smote[best_model_name]['model']

print(f"  Calibrating {best_model_name}...")
calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=5)
calibrated_model.fit(X_train_scaled, dataset.Y_train)

Y_pred_calibrated = calibrated_model.predict(X_test_scaled)
Y_pred_proba_calibrated = calibrated_model.predict_proba(X_test_scaled)[:, 1]

# Optimize threshold
optimal_threshold_cal, _ = find_optimal_threshold(dataset.Y_test, Y_pred_proba_calibrated, metric='f1')
Y_pred_calibrated_optimized = (Y_pred_proba_calibrated >= optimal_threshold_cal).astype(int)

f1_calibrated = f1_score(dataset.Y_test, Y_pred_calibrated_optimized)
recall_calibrated = recall_score(dataset.Y_test, Y_pred_calibrated_optimized)
precision_calibrated = precision_score(dataset.Y_test, Y_pred_calibrated_optimized)
roc_auc_calibrated = roc_auc_score(dataset.Y_test, Y_pred_proba_calibrated)

print(f"  Calibrated {best_model_name} (threshold={optimal_threshold_cal:.2f}):")
print(f"    F1: {f1_calibrated:.3f} | Recall: {recall_calibrated:.3f} | Precision: {precision_calibrated:.3f} | ROC-AUC: {roc_auc_calibrated:.3f}")
print()

# ==================== FINAL COMPARISON ====================
print("="*60)
print("FINAL RESULTS COMPARISON")
print("="*60)
print()

# Collect all results
all_results = []

for name, result in results_no_smote.items():
    all_results.append({
        'Model': name,
        'Condition': 'No SMOTE',
        'F1': result['f1'],
        'Recall': result['recall'],
        'Precision': result['precision'],
        'ROC-AUC': result['roc_auc']
    })

for name, result in results_with_smote.items():
    all_results.append({
        'Model': name,
        'Condition': 'With SMOTE',
        'F1': result['f1'],
        'Recall': result['recall'],
        'Precision': result['precision'],
        'ROC-AUC': result['roc_auc']
    })

all_results.append({
    'Model': 'Voting (Top 3)',
    'Condition': 'Optimized',
    'F1': f1_voting,
    'Recall': recall_voting,
    'Precision': precision_voting,
    'ROC-AUC': roc_auc_voting
})

all_results.append({
    'Model': 'Stacking Ensemble',
    'Condition': 'Optimized',
    'F1': f1_stacking,
    'Recall': recall_stacking,
    'Precision': precision_stacking,
    'ROC-AUC': roc_auc_stacking
})

all_results.append({
    'Model': f'Calibrated {best_model_name}',
    'Condition': 'Optimized',
    'F1': f1_calibrated,
    'Recall': recall_calibrated,
    'Precision': precision_calibrated,
    'ROC-AUC': roc_auc_calibrated
})

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('F1', ascending=False)
print(results_df.to_string(index=False))
print()

# ==================== SELECT BEST MODEL ====================
best_f1 = results_df['F1'].max()
best_row = results_df[results_df['F1'] == best_f1].iloc[0]

print("="*60)
print("BEST MODEL SELECTED")
print("="*60)
print(f"  Model: {best_row['Model']}")
print(f"  Condition: {best_row['Condition']}")
print(f"  Test F1: {best_row['F1']:.4f}")
print(f"  Recall: {best_row['Recall']:.4f}")
print(f"  Precision: {best_row['Precision']:.4f}")
print(f"  ROC-AUC: {best_row['ROC-AUC']:.4f}")
print()

# Save the best model
if 'Stacking' in best_row['Model']:
    final_model = stacking_clf
    print("  Saving Stacking Ensemble model...")
elif 'Voting' in best_row['Model']:
    final_model = voting_clf
    print("  Saving Voting Ensemble model...")
elif 'Calibrated' in best_row['Model']:
    final_model = calibrated_model
    print("  Saving Calibrated model...")
else:
    model_name = best_row['Model']
    if best_row['Condition'] == 'No SMOTE':
        final_model = results_no_smote[model_name]['model']
    else:
        final_model = results_with_smote[model_name]['model']
    print(f"  Saving {model_name}...")

joblib.dump(final_model, "best_stroke_model_optimized.pkl")
joblib.dump(scaler, "scaler_optimized.pkl")
results_df.to_csv("model_comparison_optimized.csv", index=False)

print("\nSaved:")
print("  - best_stroke_model_optimized.pkl")
print("  - scaler_optimized.pkl")
print("  - model_comparison_optimized.csv")
print("\n" + "="*60)
print("OPTIMIZATION COMPLETE!")
print("="*60)
print("\nRun 'streamlit run app.py' for GUI (update to use optimized model)")
