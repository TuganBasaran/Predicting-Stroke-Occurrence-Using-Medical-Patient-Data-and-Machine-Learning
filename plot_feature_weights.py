
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_factory.data_factory import StrokeDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Set up visual style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10})

def plot_feature_importance():
    print("Loading dataset...")
    path_file = "data/data.csv"
    dataset = StrokeDataset(path_file, 42)

    # ==========================================
    # 1. Tree-Based Models (Label Encoded)
    # ==========================================
    print("Training Tree-based models...")
    X_train_le = dataset.X_train_le_smote
    Y_train_le = dataset.Y_train_le_smote
    
    # Feature Names for Label Encoded Data
    # Columns in train_df excluding 'stroke'
    feature_names_le = [c for c in dataset.train_df.columns if c != "stroke"]

    # Decision Tree
    dt_model = DecisionTreeClassifier(
        criterion="entropy", splitter="best", max_depth=10, min_samples_leaf=5, class_weight="balanced"
    )
    dt_model.fit(X_train_le, Y_train_le)
    dt_importances = dt_model.feature_importances_

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=None, class_weight="balanced", random_state=42
    )
    rf_model.fit(X_train_le, Y_train_le)
    rf_importances = rf_model.feature_importances_

    # ==========================================
    # 2. Linear Models (OneHot Encoded)
    # ==========================================
    print("Training Logistic Regression...")
    X_train_ohe = dataset.X_train_ohe_smote
    Y_train_ohe = dataset.Y_train_ohe_smote

    # Feature Names for OneHot Encoded Data
    # Numerical + Binary + OneHot (order from data_factory.py lines 80)
    # cat_indices_ohe logic in data_factory implies order: num, bin, ohe
    
    num_cols = ["age", "avg_glucose_level", "bmi"]
    bin_cols = ["hypertension", "heart_disease", "ever_married", "Residence_type"]
    cat_cols = ["gender", "smoking_status", "work_type"]
    
    # Get OHE feature names
    ohe_names = list(dataset.onehot_encoder.get_feature_names_out(cat_cols))
    
    feature_names_ohe = num_cols + bin_cols + ohe_names

    # Logistic Regression
    log_reg_model = LogisticRegression(max_iter=800, class_weight="balanced")
    log_reg_model.fit(X_train_ohe, Y_train_ohe)
    
    # Use absolute value of coefficients for importance
    log_importances = np.abs(log_reg_model.coef_[0])

    # ==========================================
    # 3. Save to CSV
    # ==========================================
    print("Saving feature importances to CSV...")
    
    # Create DataFrames for each model
    df_dt = pd.DataFrame({'Feature': feature_names_le, 'DT_Importance': dt_importances})
    df_rf = pd.DataFrame({'Feature': feature_names_le, 'RF_Importance': rf_importances})
    df_log = pd.DataFrame({'Feature': feature_names_ohe, 'LogReg_Importance': log_importances})
    
    # Merge Tree-based (same features)
    df_tree = pd.merge(df_dt, df_rf, on='Feature')
    
    # For LogReg, we might have different feature names if OHE expanded them.
    # However, 'age', 'bmi', etc. are common. Let's save them separately or try to align if possible.
    # To keep it simple and clean given the different encodings, we'll save a unified CSV 
    # where we list all features. Since OHE splits features, aligning rows perfectly requires care.
    # A simple approach: Save them as separate columns, filling NaN where features don't match.
    
    df_final = pd.merge(df_tree, df_log, on='Feature', how='outer')
    df_final = df_final.sort_values(by='RF_Importance', ascending=False) # Sort by RF as default
    
    csv_path = 'feature_importances.csv'
    df_final.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # ==========================================
    # 4. Visualization
    # ==========================================
    print("Generating plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # Helper to plot
    def plot_bar(ax, importances, names, title, color):
        # Create DataFrame for easier sorting
        df_imp = pd.DataFrame({'Feature': names, 'Importance': importances})
        df_imp = df_imp.sort_values(by='Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=df_imp, ax=ax, palette=color)
        ax.set_title(title)
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')

    # Plot DT
    plot_bar(axes[0], dt_importances, feature_names_le, "Decision Tree\n(Gini Importance)", "viridis")
    
    # Plot RF
    plot_bar(axes[1], rf_importances, feature_names_le, "Random Forest\n(Gini Importance)", "magma")
    
    # Plot LogReg
    plot_bar(axes[2], log_importances, feature_names_ohe, "Logistic Regression\n(Absolute Coefficient)", "coolwarm")

    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300)
    print("Done! Plot saved as feature_importance_analysis.png")

if __name__ == "__main__":
    plot_feature_importance()
