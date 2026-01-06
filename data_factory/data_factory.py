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
            test_le[col] = le.transform(test_le[col])
            self.label_encoders[col] = le
        
        X_features = [c for c in train_le.columns if c != "stroke"]
        
        self.X_train_le = train_le[X_features].values
        self.Y_train_le = train_le["stroke"].values
        self.X_test_le = test_le[X_features].values
        self.Y_test_le = test_le["stroke"].values
        
        # SMOTE (categorical indices for label encoded data)
        cat_indices = [X_features.index(c) for c in self.categorical_cols + self.binary_cols]
        smote = SMOTENC(categorical_features=cat_indices, random_state=self.seed)
        self.X_train_le_smote, self.Y_train_le_smote = smote.fit_resample(self.X_train_le, self.Y_train_le) # type: ignore
        
    def prepare_onehot_encoded(self):
        """Linear modeller için (Logistic Regression, MLP)"""
        train_ohe = self.train_df.copy()
        test_ohe = self.test_df.copy()
        
        # OneHotEncoder
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        train_cat = self.onehot_encoder.fit_transform(train_ohe[self.categorical_cols])
        test_cat = self.onehot_encoder.transform(test_ohe[self.categorical_cols])
        
        # Numerical + Binary + OneHot birleştir
        num_bin_cols = self.numerical_cols + self.binary_cols
        
        train_num = train_ohe[num_bin_cols].values
        test_num = test_ohe[num_bin_cols].values
        
        self.X_train_ohe = np.hstack([train_num, train_cat])
        self.X_test_ohe = np.hstack([test_num, test_cat])
        self.Y_train_ohe = train_ohe["stroke"].values
        self.Y_test_ohe = test_ohe["stroke"].values
        
        # Scaling (sadece numerical sütunlar)
        self.scaler_ohe = StandardScaler()
        num_cols_count = len(self.numerical_cols)
        
        self.X_train_ohe[:, :num_cols_count] = self.scaler_ohe.fit_transform(self.X_train_ohe[:, :num_cols_count])
        self.X_test_ohe[:, :num_cols_count] = self.scaler_ohe.transform(self.X_test_ohe[:, :num_cols_count])
        
        # SMOTE için categorical indices (binary + onehot encoded)
        n_num = len(self.numerical_cols)
        n_bin = len(self.binary_cols)
        n_ohe = train_cat.shape[1]
        cat_indices_ohe = list(range(n_num, n_num + n_bin + n_ohe))
        
        smote = SMOTENC(categorical_features=cat_indices_ohe, random_state=self.seed)
        self.X_train_ohe_smote, self.Y_train_ohe_smote = smote.fit_resample(self.X_train_ohe, self.Y_train_ohe)