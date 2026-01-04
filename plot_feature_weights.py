from train_models import StrokeFeatureEngineer
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load model bundle
bundle = joblib.load('best_stroke_model.pkl')
model = bundle['model']

# Get pipeline steps
if hasattr(model, 'named_steps'):
    clf = model.named_steps['clf']
    prep = model.named_steps.get('prep', None)
else:
    clf = model
    prep = None

try:
    if hasattr(clf, 'coef_'):
        coefs = clf.coef_[0]
        if prep is not None and hasattr(prep, 'get_feature_names_out'):
            feature_names = prep.get_feature_names_out()
        else:
            feature_names = [f'x{i}' for i in range(len(coefs))]

        # Teknik isimleri insan okunur isimlerle eşleştir
        readable_names = [
            'age', 'avg_glucose_level', 'bmi', 'risk_score', 'age_glucose', 'age_bmi',
            'gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
            'Residence_type', 'smoking_status', 'age_group', 'bmi_category',
            'high_glucose', 'is_elderly', 'multiple_risks'
        ]
        # Tablo: teknik isim - gerçek isim
        print("\nFeature Mapping Table:")
        for t, r in zip(feature_names, readable_names):
            print(f"{t:15} -> {r}")
        # Grafik
        plt.figure(figsize=(10,5))
        bars = plt.barh(readable_names, coefs, color='royalblue')
        plt.xlabel('Model Weight (Coefficient)')
        plt.title('Feature Importance (Logistic Regression Coefficients)')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('results/feature_importance_logreg.png')
        plt.show()
        print('Feature importance plot saved as results/feature_importance_logreg.png')
        print('Feature names:', readable_names)
    else:
        print('Classifier does not have coef_ attribute. Cannot plot weights.')
except Exception as e:
    print('Error extracting feature importances:', e)
