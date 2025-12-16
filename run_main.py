from data_factory.data_factory import StrokeDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

path_file = "data/data.csv"


# Dataset Hazırlandı
dataset = StrokeDataset(path_file, 42) # Parameters: Path_File, seed 

print('-' * 50)
print(dataset.train_df.columns) 
print(dataset.train_df.values[0])

#DECISION TREE CLASSIFIER
# Bu tarz hiperparametreler oynanabilir - şu anki parametreler arbitrary'dir 
tree_model = DecisionTreeClassifier(criterion= "entropy", splitter= "best")

tree_model.fit(dataset.X_train, dataset.Y_train)

tree_preds = tree_model.predict(dataset.X_test)

Y_test = dataset.Y_test

tree_report = classification_report(Y_test, tree_preds)

tree_conf_matrix = confusion_matrix(Y_test, tree_preds)

print('-' * 50)
print(tree_report)
print('-' * 50)

print('-' * 50)
print(tree_conf_matrix)
print('-' * 50)


#LOGISTIC REGRESSION CLASSIFIER

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(dataset.X_train)
X_test_scaled = scaler.transform(dataset.X_test)

#Logistic Regression (balanced kalıyor)
log_reg_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

log_reg_model.fit(X_train_scaled, dataset.Y_train)

# Test set probability
log_probs = log_reg_model.predict_proba(X_test_scaled)
stroke_probs = log_probs[:, 1]

# Selected threshold from validation set (kodun en altında sadece bu değerin en iyisini bulmak için)
threshold = 0.8

# 8. Threshold'a göre tahmin üret ( normalde 0.5 ile kıyaslıyordu .pred fonksiyonunda)
log_preds_thresholded = (stroke_probs >= threshold).astype(int)

log_report = classification_report(Y_test, log_preds_thresholded)
log_conf_matrix = confusion_matrix(Y_test, log_preds_thresholded)

print(f"Logistic Regression Results (threshold = {threshold})")
print('-' * 50)
print(log_report)
print('-' * 50)
print(log_conf_matrix)
print('-' * 50)


# =====================================
# VALIDATION SET ile THRESHOLD SEÇİMİ
# =====================================

print("\nVALIDATION SET THRESHOLD ANALYSIS")
print("-" * 50)

# Validation verisini scale et
X_val_scaled = scaler.transform(dataset.X_val)
Y_val = dataset.Y_val

# Validation set için olasılıkları al
val_probs = log_reg_model.predict_proba(X_val_scaled)
val_stroke_probs = val_probs[:, 1]

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

for t in thresholds:
    val_preds = (val_stroke_probs >= t).astype(int)

    report = classification_report(
        Y_val,
        val_preds,
        output_dict=True,
        zero_division=0
    )

    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]

    print(f"Threshold = {t:.1f} | Precision = {precision:.2f} | Recall = {recall:.2f} | F1 = {f1:.2f}")