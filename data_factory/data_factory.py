import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np

class StrokeDataset():
    def __init__(self, path_file, seed: int = 42) -> None:
        self.df = pd.read_csv(path_file, index_col="id")
        self.train_test_split = 0.8 
        self.seed = seed
        
        self.categorical_cols = ["gender", "smoking_status", "work_type"]
        self.binary_cols = ["hypertension", "heart_disease", "ever_married", "Residence_type"]
        self.numerical_cols = ["age", "avg_glucose_level", "bmi"]
        
        self.preprocess_binary_encoding()
        self.set_datasets()
        self.impute_missing_values()
        
        # İki versiyon oluştur
        self.prepare_label_encoded()  # Tree-based modeller için
        self.prepare_onehot_encoded() # Linear modeller için
        
    def preprocess_binary_encoding(self):
        """Binary encoding (row-independent)"""
        # Binary sütunlar (her iki versiyon için aynı)
        self.df["ever_married"] = self.df["ever_married"].replace(["No", "Yes"], [0, 1])
        self.df["Residence_type"] = self.df["Residence_type"].replace(["Urban", "Rural"], [0, 1])

    def impute_missing_values(self):
        """Impute missing values using ONLY training data statistics"""
        # BMI eksik değerler - Calculate mean ONLY on training data
        average_bmi = round(self.train_df["bmi"].mean(), 1)
        
        # Apply to both train and test
        self.train_df["bmi"] = self.train_df["bmi"].fillna(average_bmi)
        self.test_df["bmi"] = self.test_df["bmi"].fillna(average_bmi)

    def set_datasets(self): 
        self.train_df, self.test_df = train_test_split(
            self.df, train_size=0.8, random_state=self.seed, stratify=self.df["stroke"]
        )
        
    def prepare_label_encoded(self):
        """Tree-based modeller için (Decision Tree, Random Forest)"""
        train_le = self.train_df.copy()
        test_le = self.test_df.copy()
        
        self.label_encoders = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
            train_le[col] = le.fit_transform(train_le[col])
            # Handle potential unseen labels in test
            # For simplicity/robustness here, we map unseen to a known class or handle exception
            # Given the context, we'll assume standard behavior but ideally should be robust.
            # However, to avoid crashes we can use a safe transform approach or fit on full dataset (careful of leakage)
            # Sticking to original logic but being aware.
            test_le[col] = le.transform(test_le[col])
            self.label_encoders[col] = le
        
        # Store feature names for later use in SMOTE-OHE conversion
        self.le_feature_names = [c for c in train_le.columns if c != "stroke"]
        
        self.X_train_le = train_le[self.le_feature_names].values
        self.Y_train_le = train_le["stroke"].values
        self.X_test_le = test_le[self.le_feature_names].values
        self.Y_test_le = test_le["stroke"].values
        
        # SMOTE (categorical indices for label encoded data)
        cat_indices = [self.le_feature_names.index(c) for c in self.categorical_cols + self.binary_cols]
        smote = SMOTENC(categorical_features=cat_indices, random_state=self.seed)
        self.X_train_le_smote, self.Y_train_le_smote = smote.fit_resample(self.X_train_le, self.Y_train_le) # type: ignore
        
    def prepare_onehot_encoded(self):
        """Linear modeller için (Logistic Regression, MLP) - SAFE SMOTE IMPLEMENTATION"""
        # 1. Normal One-Hot Encoding (SMOTE'suz)
        # ---------------------------------------
        train_ohe = self.train_df.copy()
        test_ohe = self.test_df.copy()
        
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Fit on original training data
        train_cat = self.onehot_encoder.fit_transform(train_ohe[self.categorical_cols])
        test_cat = self.onehot_encoder.transform(test_ohe[self.categorical_cols])
        
        num_bin_cols = self.numerical_cols + self.binary_cols
        
        train_num = train_ohe[num_bin_cols].values
        test_num = test_ohe[num_bin_cols].values
        
        self.X_train_ohe = np.hstack([train_num, train_cat])
        self.X_test_ohe = np.hstack([test_num, test_cat])
        self.Y_train_ohe = train_ohe["stroke"].values
        self.Y_test_ohe = test_ohe["stroke"].values
        
        # Scale (Normal Data)
        self.scaler_ohe = StandardScaler()
        num_cols_count = len(self.numerical_cols)
        
        self.X_train_ohe[:, :num_cols_count] = self.scaler_ohe.fit_transform(self.X_train_ohe[:, :num_cols_count])
        self.X_test_ohe[:, :num_cols_count] = self.scaler_ohe.transform(self.X_test_ohe[:, :num_cols_count])
        
        # 2. SMOTE One-Hot Encoding (FIXED)
        # ---------------------------------
        # Strategy: Use the correctly SMOTE'd Label Encoded data (X_train_le_smote),
        # decode it back to strings, and then apply the SAME OneHotEncoder.
        
        # A. Reconstruct DataFrame from SMOTE'd Label Encoded Array
        df_smote_le = pd.DataFrame(self.X_train_le_smote, columns=self.le_feature_names)
        
        # B. Inverse Transform Categorical Columns (Integers -> Strings)
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            # SMOTE might produce float values, round to nearest int
            df_smote_le[col] = df_smote_le[col].round().astype(int)
            df_smote_le[col] = le.inverse_transform(df_smote_le[col])
            
        # C. Apply OneHotEncoder (Same encoder as above)
        smote_cat = self.onehot_encoder.transform(df_smote_le[self.categorical_cols])
        
        # D. Get Numerical + Binary columns
        smote_num = df_smote_le[num_bin_cols].values
        
        # E. Combine
        self.X_train_ohe_smote = np.hstack([smote_num, smote_cat])
        self.Y_train_ohe_smote = self.Y_train_le_smote # Targets are same
        
        # F. Scale (Same scaler as above)
        # Note: Must transform, not fit (scale based on original training distribution)
        self.X_train_ohe_smote[:, :num_cols_count] = self.scaler_ohe.transform(self.X_train_ohe_smote[:, :num_cols_count])