from data_factory.data_factory import StrokeDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

path_file = "data/data.csv"

# Dataset Hazırlandı
dataset = StrokeDataset(path_file, 42)

# Tree-based modeller için (Label Encoded)
X_train_le = dataset.X_train_le
X_test_le = dataset.X_test_le
Y_train = dataset.Y_train_le
Y_test = dataset.Y_test_le

# Linear modeller için (OneHot Encoded + Scaled)
X_train_ohe = dataset.X_train_ohe
X_test_ohe = dataset.X_test_ohe

# SMOTE versiyonları (ihtiyaç olursa)
X_train_le_smote = dataset.X_train_le_smote
Y_train_le_smote = dataset.Y_train_le_smote
X_train_ohe_smote = dataset.X_train_ohe_smote
Y_train_ohe_smote = dataset.Y_train_ohe_smote


print('-' * 50)
print('TRAINING AND TESTING WITHOUT SMOTE')
print('-' * 50)


# ===================== DECISION TREE (Label Encoded) =====================

print("\nDecision Tree")

tree_model = DecisionTreeClassifier(
    criterion="entropy", 
    splitter="best", 
    class_weight="balanced", 
    max_depth=10,
    min_samples_leaf=5
)

tree_model.fit(X_train_le, Y_train)  # Label Encoded veri
tree_preds = tree_model.predict(X_test_le)

print('-' * 50)
print(classification_report(Y_test, tree_preds, digits= 3))
print('-' * 50)
print(confusion_matrix(Y_test, tree_preds))
print('-' * 50)

# ROC Curve & AUC (Decision Tree)
dt_probs = tree_model.predict_proba(X_test_le)
dt_stroke_probs = dt_probs[:, 1]
dt_auc = roc_auc_score(Y_test, dt_stroke_probs)
dt_fpr, dt_tpr, _ = roc_curve(Y_test, dt_stroke_probs)

print(f"Decision Tree AUC: {dt_auc:.4f}")

plt.figure()
plt.plot(dt_fpr, dt_tpr, label=f"Decision Tree (AUC = {dt_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Decision Tree")
plt.legend()
plt.show()


# ===================== RANDOM FOREST (Label Encoded) =====================

print("\nRandom Forest Classifier")
print("-" * 50)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train_le, Y_train)  # Label Encoded veri
rf_preds = rf_model.predict(X_test_le)

print("Random Forest Results (default threshold = 0.5)")
print('-' * 50)
print(classification_report(Y_test, rf_preds, digits= 3))
print('-' * 50)
print(confusion_matrix(Y_test, rf_preds))
print('-' * 50)

# ROC Curve & AUC (Random Forest)
rf_probs = rf_model.predict_proba(X_test_le)
rf_stroke_probs = rf_probs[:, 1]
rf_auc = roc_auc_score(Y_test, rf_stroke_probs)
rf_fpr, rf_tpr, _ = roc_curve(Y_test, rf_stroke_probs)

print(f"Random Forest AUC: {rf_auc:.4f}")

plt.figure()
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.show()


# ===================== LOGISTIC REGRESSION (OneHot Encoded) =====================

print("\nLogistic Regression Classifier")

log_reg_model = LogisticRegression(
    max_iter=800,
    class_weight="balanced"
)

log_reg_model.fit(X_train_ohe, Y_train)  # OneHot Encoded veri

# Test set probability
log_probs = log_reg_model.predict_proba(X_test_ohe)
stroke_probs = log_probs[:, 1]

# Threshold ayarı
threshold = 0.5  # 0.8'den 0.5'e değiştirdim - 0.8 çok agresif
log_preds_thresholded = (stroke_probs >= threshold).astype(int)

print(f"Logistic Regression Results (threshold = {threshold})")
print('-' * 50)
print(classification_report(Y_test, log_preds_thresholded, digits= 3))
print('-' * 50)
print(confusion_matrix(Y_test, log_preds_thresholded))
print('-' * 50)

# ROC Curve & AUC (Logistic Regression)
log_auc = roc_auc_score(Y_test, stroke_probs)
log_fpr, log_tpr, _ = roc_curve(Y_test, stroke_probs)

print(f"Logistic Regression AUC: {log_auc:.4f}")

plt.figure()
plt.plot(log_fpr, log_tpr, label=f"Logistic Regression (AUC = {log_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()


# ===================== MLP CLASSIFIER (OneHot Encoded) =====================

print("\nMLP Classifier")
print("-" * 50)

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=800,
    random_state=42, 
    early_stopping=True, 
    validation_fraction=0.1
)

mlp.fit(X_train_ohe, Y_train)  # OneHot Encoded veri
mlp_preds = mlp.predict(X_test_ohe)

print("Classification Report")
print(classification_report(Y_test, mlp_preds, digits=3)) 
print("-" * 50)
print("Confusion Matrix")
print(confusion_matrix(Y_test, mlp_preds))

# ROC Curve & AUC (MLP)
mlp_probs = mlp.predict_proba(X_test_ohe)
mlp_stroke_probs = mlp_probs[:, 1]
mlp_auc = roc_auc_score(Y_test, mlp_stroke_probs)
mlp_fpr, mlp_tpr, _ = roc_curve(Y_test, mlp_stroke_probs)

print(f"\nMLP AUC: {mlp_auc:.4f}")

plt.figure()
plt.plot(mlp_fpr, mlp_tpr, label=f"MLP (AUC = {mlp_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - MLP")
plt.legend()
plt.show()


# ===========================================================================
# ===================== SMOTE İLE EĞİTİM ====================================
# ===========================================================================

print('\n' + '=' * 50)
print('TRAINING AND TESTING WITH SMOTE')
print('=' * 50)


# ===================== DECISION TREE WITH SMOTE (Label Encoded) =====================

print("\nDecision Tree (SMOTE)")

tree_model_smote = DecisionTreeClassifier(
    criterion="entropy", 
    splitter="best", 
    max_depth=10,
    min_samples_leaf=5
)

tree_model_smote.fit(X_train_le_smote, Y_train_le_smote)  # SMOTE veri
tree_preds_smote = tree_model_smote.predict(X_test_le)

print('-' * 50)
print(classification_report(Y_test, tree_preds_smote, digits=3))
print('-' * 50)
print(confusion_matrix(Y_test, tree_preds_smote))
print('-' * 50)

# ROC Curve & AUC (Decision Tree SMOTE)
dt_probs_smote = tree_model_smote.predict_proba(X_test_le)
dt_stroke_probs_smote = dt_probs_smote[:, 1]
dt_auc_smote = roc_auc_score(Y_test, dt_stroke_probs_smote)
dt_fpr_smote, dt_tpr_smote, _ = roc_curve(Y_test, dt_stroke_probs_smote)

print(f"Decision Tree (SMOTE) AUC: {dt_auc_smote:.4f}")

plt.figure()
plt.plot(dt_fpr_smote, dt_tpr_smote, label=f"Decision Tree SMOTE (AUC = {dt_auc_smote:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Decision Tree (SMOTE)")
plt.legend()
plt.show()


# ===================== RANDOM FOREST WITH SMOTE (Label Encoded) =====================

print("\nRandom Forest Classifier (SMOTE)")
print("-" * 50)

rf_model_smote = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

rf_model_smote.fit(X_train_le_smote, Y_train_le_smote)  # SMOTE veri
rf_preds_smote = rf_model_smote.predict(X_test_le)

print("Random Forest (SMOTE) Results")
print('-' * 50)
print(classification_report(Y_test, rf_preds_smote, digits=3))
print('-' * 50)
print(confusion_matrix(Y_test, rf_preds_smote))
print('-' * 50)

# ROC Curve & AUC (Random Forest SMOTE)
rf_probs_smote = rf_model_smote.predict_proba(X_test_le)
rf_stroke_probs_smote = rf_probs_smote[:, 1]
rf_auc_smote = roc_auc_score(Y_test, rf_stroke_probs_smote)
rf_fpr_smote, rf_tpr_smote, _ = roc_curve(Y_test, rf_stroke_probs_smote)

print(f"Random Forest (SMOTE) AUC: {rf_auc_smote:.4f}")

plt.figure()
plt.plot(rf_fpr_smote, rf_tpr_smote, label=f"Random Forest SMOTE (AUC = {rf_auc_smote:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest (SMOTE)")
plt.legend()
plt.show()


# ===================== LOGISTIC REGRESSION WITH SMOTE (OneHot Encoded) =====================

print("\nLogistic Regression Classifier (SMOTE)")

log_reg_model_smote = LogisticRegression(
    max_iter=800
)

log_reg_model_smote.fit(X_train_ohe_smote, Y_train_ohe_smote)  # SMOTE veri

# Test set probability
log_probs_smote = log_reg_model_smote.predict_proba(X_test_ohe)
stroke_probs_smote = log_probs_smote[:, 1]

# Threshold ayarı
threshold = 0.5
log_preds_smote = (stroke_probs_smote >= threshold).astype(int)

print(f"Logistic Regression (SMOTE) Results (threshold = {threshold})")
print('-' * 50)
print(classification_report(Y_test, log_preds_smote, digits=3))
print('-' * 50)
print(confusion_matrix(Y_test, log_preds_smote))
print('-' * 50)

# ROC Curve & AUC (Logistic Regression SMOTE)
log_auc_smote = roc_auc_score(Y_test, stroke_probs_smote)
log_fpr_smote, log_tpr_smote, _ = roc_curve(Y_test, stroke_probs_smote)

print(f"Logistic Regression (SMOTE) AUC: {log_auc_smote:.4f}")

plt.figure()
plt.plot(log_fpr_smote, log_tpr_smote, label=f"Logistic Regression SMOTE (AUC = {log_auc_smote:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression (SMOTE)")
plt.legend()
plt.show()


# ===================== MLP CLASSIFIER WITH SMOTE (OneHot Encoded) =====================

print("\nMLP Classifier (SMOTE)")
print("-" * 50)

mlp_smote = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=800,
    random_state=42, 
    early_stopping=True, 
    validation_fraction=0.1
)

mlp_smote.fit(X_train_ohe_smote, Y_train_ohe_smote)  # SMOTE veri
mlp_preds_smote = mlp_smote.predict(X_test_ohe)

print("Classification Report")
print(classification_report(Y_test, mlp_preds_smote, digits=3)) 
print("-" * 50)
print("Confusion Matrix")
print(confusion_matrix(Y_test, mlp_preds_smote))

# ROC Curve & AUC (MLP SMOTE)
mlp_probs_smote = mlp_smote.predict_proba(X_test_ohe)
mlp_stroke_probs_smote = mlp_probs_smote[:, 1]
mlp_auc_smote = roc_auc_score(Y_test, mlp_stroke_probs_smote)
mlp_fpr_smote, mlp_tpr_smote, _ = roc_curve(Y_test, mlp_stroke_probs_smote)

print(f"\nMLP (SMOTE) AUC: {mlp_auc_smote:.4f}")

plt.figure()
plt.plot(mlp_fpr_smote, mlp_tpr_smote, label=f"MLP SMOTE (AUC = {mlp_auc_smote:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - MLP (SMOTE)")
plt.legend()
plt.show()


# ===================== Comparison Plot =====================

print("\n" + "=" * 50)
print("ALL MODELS COMPARISON")
print("=" * 50)

plt.figure(figsize=(12, 5))

# SMOTE'suz
plt.subplot(1, 2, 1)
plt.plot(dt_fpr, dt_tpr, label=f"Decision Tree (AUC = {dt_auc:.2f})")
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.2f})")
plt.plot(log_fpr, log_tpr, label=f"Logistic Reg (AUC = {log_auc:.2f})")
plt.plot(mlp_fpr, mlp_tpr, label=f"MLP (AUC = {mlp_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Without SMOTE")
plt.legend(loc="lower right")

# SMOTE'lu
plt.subplot(1, 2, 2)
plt.plot(dt_fpr_smote, dt_tpr_smote, label=f"Decision Tree (AUC = {dt_auc_smote:.2f})")
plt.plot(rf_fpr_smote, rf_tpr_smote, label=f"Random Forest (AUC = {rf_auc_smote:.2f})")
plt.plot(log_fpr_smote, log_tpr_smote, label=f"Logistic Reg (AUC = {log_auc_smote:.2f})")
plt.plot(mlp_fpr_smote, mlp_tpr_smote, label=f"MLP (AUC = {mlp_auc_smote:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - With SMOTE")
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)

# ===========================================================================
# ===================== VOTING MECHANISM (MANUAL SOFT VOTING) ===============
# ===========================================================================

print('\n' + '=' * 50)
print('TRAINING AND TESTING VOTING MECHANISM')
print('=' * 50)

# ---------------------------------------------------------------------------
# 1. Voting (Without SMOTE)
# ---------------------------------------------------------------------------
print("\nVoting Classifier (Without SMOTE) - Manual Soft Voting")
print("-" * 50)

# Tahmin Olasılıkları (Probability Averaging)
# Modellerin kendi encoding'lerine göre eğitilmiş versiyonlarını kullanıyoruz.
# Tree-based -> X_test_le
# Linear/MLP -> X_test_ohe

# Olasılıkları al (Class 1 için)
p1 = tree_model.predict_proba(X_test_le)[:, 1]
p2 = rf_model.predict_proba(X_test_le)[:, 1]
p3 = log_reg_model.predict_proba(X_test_ohe)[:, 1]
p4 = mlp.predict_proba(X_test_ohe)[:, 1]

# Olasılıkların Ortalaması (Soft Voting)
voting_probs = (p1 + p2 + p3 + p4) / 4

# Tahmin (Threshold 0.5)
voting_preds = (voting_probs >= 0.5).astype(int)

print("Voting Model Results (Without SMOTE)")
print('-' * 50)
print(classification_report(Y_test, voting_preds, digits=3))
print('-' * 50)
print(confusion_matrix(Y_test, voting_preds))
print('-' * 50)

# ROC & AUC
voting_auc = roc_auc_score(Y_test, voting_probs)
voting_fpr, voting_tpr, _ = roc_curve(Y_test, voting_probs)
print(f"Voting Model (No SMOTE) AUC: {voting_auc:.4f}")

plt.figure()
plt.plot(voting_fpr, voting_tpr, label=f"Voting Model (AUC = {voting_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Voting Model (Without SMOTE)")
plt.legend()
plt.show()


# ---------------------------------------------------------------------------
# 2. Voting (With SMOTE)
# ---------------------------------------------------------------------------
print("\nVoting Classifier (With SMOTE) - Manual Soft Voting")
print("-" * 50)

# Olasılıkları al (SMOTE Modelleri)
p1_smote = tree_model_smote.predict_proba(X_test_le)[:, 1]
p2_smote = rf_model_smote.predict_proba(X_test_le)[:, 1]
p3_smote = log_reg_model_smote.predict_proba(X_test_ohe)[:, 1]
p4_smote = mlp_smote.predict_proba(X_test_ohe)[:, 1]

# Olasılıkların Ortalaması
voting_probs_smote = (p1_smote + p2_smote + p3_smote + p4_smote) / 4

# Tahmin
voting_preds_smote = (voting_probs_smote >= 0.5).astype(int)

print("Voting Model Results (With SMOTE)")
print('-' * 50)
print(classification_report(Y_test, voting_preds_smote, digits=3))
print('-' * 50)
print(confusion_matrix(Y_test, voting_preds_smote))
print('-' * 50)

# ROC & AUC
voting_auc_smote = roc_auc_score(Y_test, voting_probs_smote)
voting_fpr_smote, voting_tpr_smote, _ = roc_curve(Y_test, voting_probs_smote)
print(f"Voting Model (SMOTE) AUC: {voting_auc_smote:.4f}")

plt.figure()
plt.plot(voting_fpr_smote, voting_tpr_smote, label=f"Voting Model SMOTE (AUC = {voting_auc_smote:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Voting Model (With SMOTE)")
plt.legend()
plt.show()


# ---------------------------------------------------------------------------
# 3. Final Comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("FINAL COMPARISON (INCLUDING VOTING)")
print("=" * 50)

plt.figure(figsize=(12, 5))

# SMOTE'suz
plt.subplot(1, 2, 1)
plt.plot(dt_fpr, dt_tpr, label=f"DT ({dt_auc:.2f})")
plt.plot(rf_fpr, rf_tpr, label=f"RF ({rf_auc:.2f})")
plt.plot(log_fpr, log_tpr, label=f"LR ({log_auc:.2f})")
plt.plot(mlp_fpr, mlp_tpr, label=f"MLP ({mlp_auc:.2f})")
plt.plot(voting_fpr, voting_tpr, 'k--', linewidth=2, label=f"Voting ({voting_auc:.2f})") # Voting siyah kesik çizgi
plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curves - Without SMOTE")
plt.legend(loc="lower right", fontsize='small')

# SMOTE'lu
plt.subplot(1, 2, 2)
plt.plot(dt_fpr_smote, dt_tpr_smote, label=f"DT ({dt_auc_smote:.2f})")
plt.plot(rf_fpr_smote, rf_tpr_smote, label=f"RF ({rf_auc_smote:.2f})")
plt.plot(log_fpr_smote, log_tpr_smote, label=f"LR ({log_auc_smote:.2f})")
plt.plot(mlp_fpr_smote, mlp_tpr_smote, label=f"MLP ({mlp_auc_smote:.2f})")
plt.plot(voting_fpr_smote, voting_tpr_smote, 'k--', linewidth=2, label=f"Voting ({voting_auc_smote:.2f})")
plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curves - With SMOTE")
plt.legend(loc="lower right", fontsize='small')

plt.tight_layout()
plt.show()

print(f"{'Model':<25} {'No SMOTE':<15} {'SMOTE':<15}")
print("-" * 55)
print(f"{'Decision Tree':<25} {dt_auc:<15.4f} {dt_auc_smote:<15.4f}")
print(f"{'Random Forest':<25} {rf_auc:<15.4f} {rf_auc_smote:<15.4f}")
print(f"{'Logistic Regression':<25} {log_auc:<15.4f} {log_auc_smote:<15.4f}")
print(f"{'MLP':<25} {mlp_auc:<15.4f} {mlp_auc_smote:<15.4f}")
print(f"{'Voting (Soft)':<25} {voting_auc:<15.4f} {voting_auc_smote:<15.4f}")


