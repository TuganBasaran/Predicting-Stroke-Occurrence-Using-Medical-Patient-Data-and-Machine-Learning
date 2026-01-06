"""
Stroke Prediction (Leakage-Free) — Single Script (FINAL)
========================================================
Fixes:
1) SMOTENC is applied INSIDE each CV fold (no leakage)
2) Feature engineering + scaling are INSIDE CV (no leakage)
3) Threshold optimization uses VALIDATION set (not test)
4) Ensembling uses CLONE() to avoid estimator state sharing
5) Robust categorical handling: if raw categorical columns come as strings,
   they are encoded deterministically inside the transformer (fit-time mapping),
   so SMOTENC always sees non-negative ints.

Outputs:
- model_comparison_results.csv
- kfold_cv_detailed_results.csv
- best_stroke_model.pkl  (fitted on train+val, never using test)

Assumptions:
- from data_factory.data_factory import StrokeDataset
- StrokeDataset exposes: X_train, X_val, X_test, Y_train, Y_val, Y_test
- X arrays have 10 base features in this order:
  ['gender','age','hypertension','heart_disease','ever_married','work_type',
   'Residence_type','avg_glucose_level','bmi','smoking_status']
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from data_factory.data_factory import StrokeDataset

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline


# ---------------------------
# 0) Config
# ---------------------------
SEED = 42
K_FOLDS = 10

DEFAULT_PATH = "data/data.csv"
FALLBACK_PATH = "/mnt/data/data.csv"
path_file = DEFAULT_PATH if os.path.exists(DEFAULT_PATH) else FALLBACK_PATH

feature_names = [
    "gender", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"
]

# Original categorical indices in the 10 raw features
RAW_CAT_IDX = [0, 2, 3, 4, 5, 6, 9]


# ---------------------------
# 1) Feature Engineering Transformer
# ---------------------------
class StrokeFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Input:  (n,10) in the 'feature_names' order
    Output: (n,18) with engineered features appended:

      0..9   : original
      10     : age_group (cat)
      11     : bmi_category (cat)
      12     : risk_score (ordinal int)
      13     : high_glucose (binary/cat)
      14     : is_elderly (binary/cat)
      15     : multiple_risks (binary/cat)
      16     : age_glucose (continuous)
      17     : age_bmi (continuous)
    """

    def __init__(self, feature_names_=None, raw_cat_cols=None):
        self.feature_names_ = feature_names_
        self.raw_cat_cols = raw_cat_cols

    def _fit_cat_mapping(self, series: pd.Series):
        # Build a stable mapping for categories -> non-negative int
        # Unknowns at transform-time will map to 0.
        s = series.astype("string").fillna("UNK").str.strip()
        cats = pd.Index(pd.unique(s))
        # ensure "UNK" maps to 0
        if "UNK" in cats:
            cats = pd.Index(["UNK"] + [c for c in cats if c != "UNK"])
        mapping = {cat: i for i, cat in enumerate(cats)}
        return mapping

    def _apply_cat_mapping(self, series: pd.Series, mapping: dict):
        s = series.astype("string").fillna("UNK").str.strip()
        out = s.map(mapping).fillna(0).astype(int)
        # SMOTENC expects non-negative ints
        out[out < 0] = 0
        return out

    def fit(self, X, y=None):
        if self.feature_names_ is None:
            raise ValueError("feature_names_ must be provided")
        
        self.feature_names_list_ = list(self.feature_names_)
        self.raw_cat_cols_list_ = list(self.raw_cat_cols) if self.raw_cat_cols is not None else []

        X_df = pd.DataFrame(X, columns=self.feature_names_list_)

        # medians for NA handling (numeric)
        self.bmi_median_ = pd.to_numeric(X_df["bmi"], errors="coerce").median()
        self.age_median_ = pd.to_numeric(X_df["age"], errors="coerce").median()
        self.glucose_median_ = pd.to_numeric(X_df["avg_glucose_level"], errors="coerce").median()

        # fit mappings for raw categorical cols (if they are strings, we encode)
        self.cat_mappings_ = {}
        for col in self.raw_cat_cols_list_:
            self.cat_mappings_[col] = self._fit_cat_mapping(X_df[col])

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names_list_).copy()

        # numeric coercion + NA fill
        X_df["bmi"] = pd.to_numeric(X_df["bmi"], errors="coerce").fillna(self.bmi_median_)
        X_df["age"] = pd.to_numeric(X_df["age"], errors="coerce").fillna(self.age_median_)
        X_df["avg_glucose_level"] = pd.to_numeric(X_df["avg_glucose_level"], errors="coerce").fillna(self.glucose_median_)

        # encode raw categoricals robustly (handles if already numeric too)
        for col in self.raw_cat_cols_list_:
            # if it's already numeric-ish, keep numeric; else map with fitted mapping
            s = X_df[col]
            numeric = pd.to_numeric(s, errors="coerce")
            if numeric.notna().mean() > 0.95:  # mostly numeric
                X_df[col] = numeric.fillna(0).astype(int)
                X_df[col] = X_df[col].clip(lower=0)
            else:
                X_df[col] = self._apply_cat_mapping(s, self.cat_mappings_[col])

        # ensure binary numeric cols are int (hypertension, heart_disease often are already 0/1)
        for col in ["hypertension", "heart_disease", "ever_married"]:
            if col in X_df.columns:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0).astype(int).clip(lower=0)

        # engineered categorical bins
        age_bins = [0, 30, 50, 70, 200]
        X_df["age_group"] = pd.cut(X_df["age"], bins=age_bins, include_lowest=True).cat.codes.astype(int)
        X_df.loc[X_df["age_group"] < 0, "age_group"] = 0

        bmi_bins = [0, 18.5, 25, 30, 1000]
        X_df["bmi_category"] = pd.cut(X_df["bmi"], bins=bmi_bins, include_lowest=True).cat.codes.astype(int)
        X_df.loc[X_df["bmi_category"] < 0, "bmi_category"] = 0

        X_df["high_glucose"] = (X_df["avg_glucose_level"] > 140).astype(int)
        X_df["is_elderly"] = (X_df["age"] > 65).astype(int)
        X_df["multiple_risks"] = ((X_df["hypertension"].astype(int) + X_df["heart_disease"].astype(int)) >= 2).astype(int)

        # risk_score as ORDINAL INT (kept numeric, not categorical)
        X_df["risk_score"] = (
            (X_df["age"] > 60).astype(int)
            + X_df["hypertension"].astype(int)
            + X_df["heart_disease"].astype(int)
            + (X_df["avg_glucose_level"] > 140).astype(int)
            + (X_df["bmi"] > 30).astype(int)
        ).astype(int)

        # interactions
        X_df["age_glucose"] = X_df["age"] * X_df["avg_glucose_level"] / 100.0
        X_df["age_bmi"] = X_df["age"] * X_df["bmi"] / 100.0

        out_cols = self.feature_names_list_ + [
            "age_group", "bmi_category", "risk_score",
            "high_glucose", "is_elderly", "multiple_risks",
            "age_glucose", "age_bmi"
        ]
        return X_df[out_cols].values


# ---------------------------
# 2) Preprocessor (scale numeric only)
# ---------------------------
# After engineering -> 18 features
# Categorical features in engineered space:
# raw cats + engineered cats (age_group, bmi_category, high_glucose, is_elderly, multiple_risks)
ENG_CAT_IDX = RAW_CAT_IDX + [10, 11, 13, 14, 15]
ENG_NUM_IDX = [i for i in range(18) if i not in ENG_CAT_IDX]  # includes risk_score as numeric

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ENG_NUM_IDX),
        ("cat", "passthrough", ENG_CAT_IDX),
    ],
    remainder="drop",
)

# After preprocessor: numeric first, categorical appended
SMOTE_CAT_IDX_AFTER_PREP = list(range(len(ENG_NUM_IDX), len(ENG_NUM_IDX) + len(ENG_CAT_IDX)))


# ---------------------------
# 3) Pipelines
# ---------------------------
def make_pipelines(seed: int):
    lr = LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced")
    dt = DecisionTreeClassifier(random_state=seed, class_weight="balanced")
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=-1)
    gb = GradientBoostingClassifier(random_state=seed)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=800,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1
    )

    # Feature engineer will robustly encode raw categorical columns if needed
    fe = StrokeFeatureEngineer(feature_names_=feature_names, raw_cat_cols=[
        feature_names[i] for i in RAW_CAT_IDX
    ])

    no_smote = {
        "Logistic Regression": ImbPipeline([("fe", clone(fe)), ("prep", preprocessor), ("clf", lr)]),
        "MLP": ImbPipeline([("fe", clone(fe)), ("prep", preprocessor), ("clf", mlp)]),
        "Decision Tree": ImbPipeline([("fe", clone(fe)), ("clf", dt)]),
        "Random Forest": ImbPipeline([("fe", clone(fe)), ("clf", rf)]),
        "Gradient Boosting": ImbPipeline([("fe", clone(fe)), ("clf", gb)]),
    }

    # SMOTE inside CV folds
    smote_after_prep = SMOTENC(
        categorical_features=SMOTE_CAT_IDX_AFTER_PREP,
        random_state=seed,
        k_neighbors=5,
        sampling_strategy=0.3
    )

    smote_on_eng = SMOTENC(
        categorical_features=ENG_CAT_IDX,
        random_state=seed,
        k_neighbors=5,
        sampling_strategy=0.3
    )

    with_smote = {
        "Logistic Regression": ImbPipeline([
            ("fe", clone(fe)),
            ("prep", preprocessor),
            ("smote", smote_after_prep),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed))
        ]),
        "MLP": ImbPipeline([
            ("fe", clone(fe)),
            ("prep", preprocessor),
            ("smote", smote_after_prep),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=800,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.1
            ))
        ]),
        "Decision Tree": ImbPipeline([("fe", clone(fe)), ("smote", smote_on_eng), ("clf", DecisionTreeClassifier(random_state=seed))]),
        "Random Forest": ImbPipeline([("fe", clone(fe)), ("smote", smote_on_eng), ("clf", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1))]),
        "Gradient Boosting": ImbPipeline([("fe", clone(fe)), ("smote", smote_on_eng), ("clf", GradientBoostingClassifier(random_state=seed))]),
    }

    return no_smote, with_smote


# ---------------------------
# 4) CV helper
# ---------------------------
def evaluate_cv(pipeline, X, y, k_folds: int, seed: int):
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    scoring = {"f1": "f1", "recall": "recall", "precision": "precision", "roc_auc": "roc_auc"}
    return cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)


# ---------------------------
# 5) Threshold selection on validation
# ---------------------------
def find_best_threshold(estimator, X_val, y_val, metric="f1"):
    proba = estimator.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.05, 0.95, 0.05)
    best_t, best_s = 0.5, -1.0

    for t in thresholds:
        pred = (proba >= t).astype(int)
        if metric == "f1":
            s = f1_score(y_val, pred)
        elif metric == "recall":
            s = recall_score(y_val, pred)
        else:
            raise ValueError("metric must be 'f1' or 'recall'")

        if s > best_s:
            best_s, best_t = s, t

    return float(best_t), float(best_s)


def eval_on_test(estimator, X_test, y_test, threshold=0.5):
    proba = estimator.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "F1": float(f1_score(y_test, pred)),
        "Recall": float(recall_score(y_test, pred)),
        "Precision": float(precision_score(y_test, pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, proba)),
    }


# ---------------------------
# 6) Main
# ---------------------------
def main():
    print("\nStroke Prediction (Leakage-Free) — FINAL")
    print(f"Data path: {path_file}")
    print(f"K-Fold CV: {K_FOLDS} | Seed: {SEED}\n")

    dataset = StrokeDataset(path_file, SEED)
    X_train, y_train = dataset.X_train, np.array(dataset.Y_train).astype(int)
    X_val, y_val = dataset.X_val, np.array(dataset.Y_val).astype(int)
    X_test, y_test = dataset.X_test, np.array(dataset.Y_test).astype(int)

    print("Dataset Info:")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Train class counts: No Stroke={np.sum(y_train==0)} | Stroke={np.sum(y_train==1)}")
    if np.sum(y_train == 1) > 0:
        print(f"  Imbalance ratio: {np.sum(y_train==0)/np.sum(y_train==1):.1f}:1\n")

    no_smote_pipes, with_smote_pipes = make_pipelines(SEED)

    detailed_rows = []
    summary_rows = []

    for condition_name, pipes in [("No SMOTE", no_smote_pipes), ("With SMOTE", with_smote_pipes)]:
        print(f"=== CV: {condition_name} ===")
        for model_name, pipe in pipes.items():
            cv_out = evaluate_cv(pipe, X_train, y_train, K_FOLDS, SEED)

            f1_scores = cv_out["test_f1"]
            rec_scores = cv_out["test_recall"]
            pre_scores = cv_out["test_precision"]
            auc_scores = cv_out["test_roc_auc"]

            print(
                f"  {model_name:<18} | "
                f"F1={f1_scores.mean():.3f}±{f1_scores.std():.3f} "
                f"Recall={rec_scores.mean():.3f}±{rec_scores.std():.3f} "
                f"AUC={auc_scores.mean():.3f}±{auc_scores.std():.3f}"
            )

            for fold_i in range(len(f1_scores)):
                detailed_rows.append({
                    "Model": model_name,
                    "Condition": condition_name,
                    "Fold": fold_i + 1,
                    "F1": float(f1_scores[fold_i]),
                    "Recall": float(rec_scores[fold_i]),
                    "Precision": float(pre_scores[fold_i]),
                    "ROC_AUC": float(auc_scores[fold_i]),
                })

            summary_rows.append({
                "Model": model_name,
                "Condition": condition_name,
                "CV_F1_Mean": float(f1_scores.mean()),
                "CV_F1_Std": float(f1_scores.std()),
                "CV_Recall_Mean": float(rec_scores.mean()),
                "CV_Recall_Std": float(rec_scores.std()),
                "CV_Precision_Mean": float(pre_scores.mean()),
                "CV_Precision_Std": float(pre_scores.std()),
                "CV_ROC_AUC_Mean": float(auc_scores.mean()),
                "CV_ROC_AUC_Std": float(auc_scores.std()),
            })
        print()

    pd.DataFrame(detailed_rows).to_csv("kfold_cv_detailed_results.csv", index=False)
    print("Saved: kfold_cv_detailed_results.csv")

    # Fit on TRAIN, tune threshold on VAL, report TEST
    results_rows = []

    def fit_and_score(condition_label, pipelines_dict):
        nonlocal results_rows
        for model_name, pipe in pipelines_dict.items():
            pipe.fit(X_train, y_train)
            t_f1, val_f1 = find_best_threshold(pipe, X_val, y_val, metric="f1")
            test_metrics = eval_on_test(pipe, X_test, y_test, threshold=t_f1)

            results_rows.append({
                "Model": model_name,
                "Condition": condition_label,
                "Threshold": t_f1,
                "Val_F1": val_f1,
                **test_metrics
            })

    print("\n=== Test Evaluation (threshold tuned on VAL, reported on TEST) ===")
    fit_and_score("No SMOTE", no_smote_pipes)
    fit_and_score("With SMOTE", with_smote_pipes)

    results_df = pd.DataFrame(results_rows)

    # Ensembles: pick top-3 from CV mean F1 (No SMOTE)
    cv_summary_df = pd.DataFrame(summary_rows)
    cv_no = cv_summary_df[cv_summary_df["Condition"] == "No SMOTE"].sort_values("CV_F1_Mean", ascending=False)
    top3_names = cv_no["Model"].head(3).tolist()
    print(f"\nTop-3 base models by CV F1 (No SMOTE): {top3_names}")

    estimators_top3 = [(name, clone(no_smote_pipes[name])) for name in top3_names]

    voting = VotingClassifier(estimators=estimators_top3, voting="soft", n_jobs=-1)
    voting.fit(X_train, y_train)
    t_vote, val_f1_vote = find_best_threshold(voting, X_val, y_val, metric="f1")
    vote_metrics = eval_on_test(voting, X_test, y_test, threshold=t_vote)
    results_df = pd.concat([results_df, pd.DataFrame([{
        "Model": "Voting (Top 3)",
        "Condition": "Optimized",
        "Threshold": t_vote,
        "Val_F1": val_f1_vote,
        **vote_metrics
    }])], ignore_index=True)

    stacking = StackingClassifier(
        estimators=estimators_top3,
        final_estimator=LogisticRegression(max_iter=2000, random_state=SEED),
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1
    )
    stacking.fit(X_train, y_train)
    t_stack, val_f1_stack = find_best_threshold(stacking, X_val, y_val, metric="f1")
    stack_metrics = eval_on_test(stacking, X_test, y_test, threshold=t_stack)
    results_df = pd.concat([results_df, pd.DataFrame([{
        "Model": "Stacking Ensemble",
        "Condition": "Optimized",
        "Threshold": t_stack,
        "Val_F1": val_f1_stack,
        **stack_metrics
    }])], ignore_index=True)

    best_single_name = cv_no["Model"].iloc[0]
    calibrated = CalibratedClassifierCV(clone(no_smote_pipes[best_single_name]), method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    t_cal, val_f1_cal = find_best_threshold(calibrated, X_val, y_val, metric="f1")
    cal_metrics = eval_on_test(calibrated, X_test, y_test, threshold=t_cal)
    results_df = pd.concat([results_df, pd.DataFrame([{
        "Model": f"Calibrated {best_single_name}",
        "Condition": "Optimized",
        "Threshold": t_cal,
        "Val_F1": val_f1_cal,
        **cal_metrics
    }])], ignore_index=True)

    # Model selection: sort by VAL F1 (not by test)
    results_df = results_df.sort_values("Val_F1", ascending=False).reset_index(drop=True)
    results_df.to_csv("model_comparison_results.csv", index=False)
    print("Saved: model_comparison_results.csv\n")

    print("=== FINAL RANKING (Sorted by Validation F1) ===")
    print(results_df[["Model", "Condition", "Threshold", "Val_F1", "F1", "Recall", "Precision", "ROC-AUC"]].to_string(index=False))

    # Save best model refit on TRAIN+VAL (never use TEST)
    best_row = results_df.iloc[0]
    best_model_name = best_row["Model"]
    best_condition = best_row["Condition"]
    best_threshold = float(best_row["Threshold"])

    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    if best_model_name == "Voting (Top 3)":
        estimators_tv = []
        for name in top3_names:
            est = clone(no_smote_pipes[name])
            est.fit(X_trainval, y_trainval)
            estimators_tv.append((name, est))
        final_estimator = VotingClassifier(estimators=estimators_tv, voting="soft", n_jobs=-1)
        final_estimator.fit(X_trainval, y_trainval)

    elif best_model_name == "Stacking Ensemble":
        estimators_ts = []
        for name in top3_names:
            est = clone(no_smote_pipes[name])
            est.fit(X_trainval, y_trainval)
            estimators_ts.append((name, est))
        final_estimator = StackingClassifier(
            estimators=estimators_ts,
            final_estimator=LogisticRegression(max_iter=2000, random_state=SEED),
            stack_method="predict_proba",
            cv=5,
            n_jobs=-1
        )
        final_estimator.fit(X_trainval, y_trainval)

    elif best_model_name.startswith("Calibrated "):
        final_estimator = CalibratedClassifierCV(clone(no_smote_pipes[best_single_name]), method="sigmoid", cv=5)
        final_estimator.fit(X_trainval, y_trainval)

    else:
        if best_condition == "No SMOTE":
            final_estimator = clone(no_smote_pipes[best_model_name])
        else:
            final_estimator = clone(with_smote_pipes[best_model_name])
        final_estimator.fit(X_trainval, y_trainval)

    bundle = {
        "model": final_estimator,
        "threshold": best_threshold,
        "meta": {
            "best_row": best_row.to_dict(),
            "feature_names_raw": feature_names,
            "seed": SEED,
            "k_folds": K_FOLDS,
        }
    }

    joblib.dump(bundle, "best_stroke_model.pkl")
    print("\nSaved: best_stroke_model.pkl")
    print(f"Best model saved: {best_model_name} | Condition: {best_condition} | Threshold: {best_threshold:.2f}")
    print("Done.")


if __name__ == "__main__":
    main()