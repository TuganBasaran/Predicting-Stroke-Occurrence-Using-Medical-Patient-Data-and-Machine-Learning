import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report
)
from matplotlib.colors import LogNorm

# IMPORTANT: Ensure custom transformer is importable so joblib can load the pipeline
from train_models import StrokeFeatureEngineer  # noqa: F401

from data_factory.data_factory import StrokeDataset


# ---------------------------
# Config
# ---------------------------
SEED = 42
PATH_FILE = "data/data.csv"
MODEL_PATH = "best_stroke_model.pkl"
RESULTS_DIR = "results"

sns.set_theme(style="white", context="talk")
custom_palette = ["#3498db", "#e74c3c"]


# These MUST match train_models.py
RAW_FEATURE_NAMES = [
    "gender", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"
]

RAW_CAT_IDX = [0, 2, 3, 4, 5, 6, 9]

# Engineered feature order is: original 10 + appended 8 (same as StrokeFeatureEngineer)
ENG_FEATURE_NAMES = RAW_FEATURE_NAMES + [
    "age_group", "bmi_category", "risk_score",
    "high_glucose", "is_elderly", "multiple_risks",
    "age_glucose", "age_bmi"
]

# After engineering -> 18 features
ENG_CAT_IDX = RAW_CAT_IDX + [10, 11, 13, 14, 15]
ENG_NUM_IDX = [i for i in range(18) if i not in ENG_CAT_IDX]


# ---------------------------
# Load
# ---------------------------
def load_data_and_model():
    print("Loading data...")
    dataset = StrokeDataset(PATH_FILE, SEED)

    X_test = dataset.X_test
    y_test = np.array(dataset.Y_test).astype(int)

    print(f"Loading model from {MODEL_PATH}...")
    bundle = joblib.load(MODEL_PATH)

    return X_test, y_test, bundle


# ---------------------------
# Plots
# ---------------------------
def plot_confusion_matrix(y_test, y_pred, threshold):
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate row percentages (within actual class)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_perc = cm / np.maximum(row_sums, 1) * 100.0

    # Table-style confusion matrix (always visible, readable)
    cell_text = [
        [
            f"True Negative\n(Healthy)\nCount: {tn}\n{row_perc[0,0]:.1f}% of Healthy",
            f"False Positive\n(False Alarm)\nCount: {fp}\n{row_perc[0,1]:.1f}% of Healthy"
        ],
        [
            f"False Negative\n(Missed Stroke)\nCount: {fn}\n{row_perc[1,0]:.1f}% of Stroke",
            f"True Positive\n(Correct Stroke)\nCount: {tp}\n{row_perc[1,1]:.1f}% of Stroke"
        ]
    ]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    table = ax.table(
        cellText=cell_text,
        rowLabels=["Actual Healthy", "Actual Stroke"],
        colLabels=["Predicted Healthy", "Predicted Stroke"],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.3, 2.3)
    ax.axis('off')
    # Professional coloring: header dark, alternating row colors, highlight max cell
    max_val = max(tn, fp, fn, tp)
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        elif cell_text[row-1][col] == cell_text[(tn, fp, fn, tp).index(max_val)//2][(tn, fp, fn, tp).index(max_val)%2]:
            cell.set_facecolor('#ffe082')  # gold highlight for max cell
            cell.set_text_props(weight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f5f6fa')  # light gray
        else:
            cell.set_facecolor('#e1e8ed')  # slightly darker gray
    plt.title(f"Confusion Matrix (Threshold={threshold:.2f})", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_test, y_prob, threshold):
    print("Generating ROC Curve...")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    idx = (np.abs(thresholds - threshold)).argmin()
    current_fpr = fpr[idx]
    current_tpr = tpr[idx]

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color=custom_palette[1], lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    plt.scatter(current_fpr, current_tpr, s=100, c="red", marker="o",
                label=f"Selected Threshold ({threshold:.2f})")
    plt.annotate(
        f"Threshold: {threshold:.2f}\nTPR: {current_tpr:.2f}\nFPR: {current_fpr:.2f}",
        (current_fpr, current_tpr),
        xytext=(min(current_fpr + 0.1, 0.9), max(current_tpr - 0.1, 0.1)),
        arrowprops=dict(
            facecolor="#34495e",
            edgecolor="#34495e",
            linewidth=2.5,
            arrowstyle="->",
            shrinkA=8,
            shrinkB=8,
            alpha=0.85,
            connectionstyle="arc3,rad=0.2"
        ),
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f6fa", ec="#34495e", lw=1.5, alpha=0.95),
        fontsize=12,
        fontweight='bold',
        color="#222"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity/Recall)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=300)
    plt.close()


def plot_precision_recall_curve(y_test, y_prob, threshold):
    print("Generating Precision-Recall Curve...")
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    # thresholds length = n, precision/recall length = n+1
    if len(thresholds) == 0:
        print("PR curve thresholds empty; skipping.")
        return

    idx = (np.abs(thresholds - threshold)).argmin()
    current_prec = precision[idx]
    current_rec = recall[idx]

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color=custom_palette[0], lw=2)

    plt.scatter(current_rec, current_prec, s=100, c="red", marker="o",
                label=f"Selected Threshold ({threshold:.2f})")
    plt.annotate(
        f"Threshold: {threshold:.2f}\nPrecision: {current_prec:.2f}\nRecall: {current_rec:.2f}",
        (current_rec, current_prec),
        xytext=(max(current_rec - 0.2, 0.05), max(current_prec - 0.1, 0.05)),
        arrowprops=dict(
            facecolor="#34495e",
            edgecolor="#34495e",
            linewidth=2.5,
            arrowstyle="->",
            shrinkA=8,
            shrinkB=8,
            alpha=0.85,
            connectionstyle="arc3,rad=0.2"
        ),
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f6fa", ec="#34495e", lw=1.5, alpha=0.95),
        fontsize=12,
        fontweight='bold',
        color="#222"
    )

    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_curve.png"), dpi=300)
    plt.close()


def _get_classifier_from_pipeline(pipeline):
    # Your training pipelines used step name "clf"
    if hasattr(pipeline, "named_steps"):
        if "clf" in pipeline.named_steps:
            return pipeline.named_steps["clf"]
        if "classifier" in pipeline.named_steps:
            return pipeline.named_steps["classifier"]
    # fallback: last step
    try:
        return pipeline.steps[-1][1]
    except Exception:
        return None


def _infer_feature_names_for_importance(model):
    """
    Returns feature names aligned with coefficient/feature_importances_ length.

    Cases:
    1) Pipeline includes "prep": output = scaled numeric (ENG_NUM_IDX) + categorical passthrough (ENG_CAT_IDX)
    2) No "prep": output = engineered 18 features
    """
    if hasattr(model, "named_steps") and "prep" in model.named_steps:
        num_names = [ENG_FEATURE_NAMES[i] for i in ENG_NUM_IDX]
        cat_names = [ENG_FEATURE_NAMES[i] for i in ENG_CAT_IDX]
        # same ordering as ColumnTransformer: num first, then cat passthrough
        return [f"num__{n}" for n in num_names] + [f"cat__{n}" for n in cat_names]
    else:
        return ENG_FEATURE_NAMES


def plot_feature_importance(model):
    print("Generating Feature Importance...")
    # Try to extract feature importances or coefficients
    importances = None
    feature_names = _infer_feature_names_for_importance(model)
    classifier = _get_classifier_from_pipeline(model)
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    if importances is not None and feature_names is not None:
        # Human-friendly feature name mapping
        pretty_names = {
            "gender": "Gender",
            "age": "Age",
            "hypertension": "Hypertension",
            "heart_disease": "Heart Disease",
            "ever_married": "Married",
            "work_type": "Work Type",
            "Residence_type": "Residence Type",
            "avg_glucose_level": "Avg Glucose",
            "bmi": "BMI",
            "smoking_status": "Smoking Status",
            "age_group": "Age Group",
            "bmi_category": "BMI Category",
            "risk_score": "Risk Score",
            "high_glucose": "High Glucose",
            "is_elderly": "Elderly",
            "multiple_risks": "Multiple Risks",
            "age_glucose": "Age × Glucose",
            "age_bmi": "Age × BMI"
        }
        def clean_name(f):
            # Remove any 'cat_' or similar technical prefix
            f_clean = f
            if f_clean.startswith('cat_'):
                f_clean = f_clean[4:]
            return pretty_names.get(f_clean, f_clean.replace('_', ' ').title())
        feat_names_pretty = [clean_name(f) for f in feature_names]
        fi_df = pd.DataFrame({"Feature": feat_names_pretty, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=False).head(15)
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis")
        plt.title("Top 15 Feature Importances / |Coefficients|", fontsize=15)
        plt.xlabel("Importance", fontsize=13)
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=300)
        plt.close()
    else:
        print("Model does not support feature importance extraction or feature names unavailable.")


def plot_model_comparison_table(csv_path="model_comparison_results.csv", save_path="results/model_comparison_table.png"):
    """Plot a table comparing all models and their metrics with professional coloring and better fit for long names."""
    df = pd.read_csv(csv_path)
    # Optionally round metrics for readability
    metric_cols = ["Threshold", "F1", "Recall", "Precision", "ROC-AUC"]
    df[metric_cols] = df[metric_cols].round(3)
    # Wrap long model names for better fit
    df["Model"] = df["Model"].apply(lambda x: '\n'.join(x.split(' ')) if len(x) > 18 else x)
    # Select and reorder columns for display
    display_df = df[[
        "Model", "Condition", "Threshold", "F1", "Recall", "Precision", "ROC-AUC"
    ]].rename(columns={
        "Model": "Model",
        "Condition": "Condition",
        "Threshold": "Thresh.",
        "F1": "Test F1",
        "Recall": "Recall",
        "Precision": "Precision",
        "ROC-AUC": "ROC-AUC"
    })
    n_rows, n_cols = display_df.shape
    fig_width = min(2 + n_cols * 2, 18)
    fig_height = 0.7 * n_rows + 2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    # Shrink font if too many rows
    font_size = 12 if n_rows <= 15 else 10
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.1, 1.1)
    # Professional coloring: header dark, alternating row colors, highlight best Test F1
    best_f1_idx = display_df["Test F1"].astype(float).idxmax() + 1  # +1 for header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        elif row == best_f1_idx:
            cell.set_facecolor('#ffe082')  # gold highlight for best Test F1
            cell.set_text_props(weight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f5f6fa')  # light gray
        else:
            cell.set_facecolor('#e1e8ed')  # slightly darker gray
    # Auto-adjust column widths for long names
    for i, col in enumerate(display_df.columns):
        maxlen = max([len(str(x)) for x in display_df[col]] + [len(col)])
        table.auto_set_column_width(i)
    plt.title("Model Comparison Table", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ---------------------------
# Main
# ---------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_test, y_test, bundle = load_data_and_model()

    model = bundle["model"]
    threshold = float(bundle["threshold"])
    meta = bundle.get("meta", {})
    best_row = meta.get("best_row", {})

    print(f"Model: {best_row.get('Model', 'Unknown')}")
    print(f"Condition: {best_row.get('Condition', 'Unknown')}")
    print(f"Optimized Threshold: {threshold}")

    print("Predicting...")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    plot_confusion_matrix(y_test, y_pred, threshold)
    plot_roc_curve(y_test, y_prob, threshold)
    plot_precision_recall_curve(y_test, y_prob, threshold)

    # Feature importance / coefficients
    try:
        plot_feature_importance(model)
    except Exception as e:
        print(f"Could not generate feature importance: {e}")

    plot_model_comparison_table()

    print("\nVisualization generation complete! Check the 'results' folder.")


if __name__ == "__main__":
    main()