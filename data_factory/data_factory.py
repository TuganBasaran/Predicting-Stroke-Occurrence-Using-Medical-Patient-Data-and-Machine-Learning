import pandas as pd
from sklearn.model_selection import train_test_split

class StrokeDataset():
    def __init__(self, path_file, seed: int = 42) -> None:
        self.df = pd.read_csv(path_file, index_col= "id")
        self.train_test_split = 0.8 
        self.seed = seed
        self.label_dataset()
        self.set_datasets()

        self.X_features = self.df.columns.to_list()[0: -1]

        self.X_train = self.train_df[self.X_features].values
        self.Y_train = self.train_df["stroke"].values

        self.X_val = self.val_df[self.X_features].values
        self.Y_val = self.val_df["stroke"].values

        self.X_test = self.test_df[self.X_features].values
        self.Y_test = self.test_df["stroke"].values
        
        

    def label_dataset(self): 
        #TODO: Numerical sayılar StandardScaler kullanılarak çözülebilir 
        # TODO: Median'da kullanılabilir 
        # BMI'da eksik data'lar var. Eksik data'ları average BMI doldur - 
        average_bmi = round(self.df["bmi"].mean(), 1)
        self.df["bmi"] = self.df["bmi"].fillna(average_bmi)

        # Sigara içme durumlarını label'la
        self.df["smoking_status"] = self.df["smoking_status"].replace(["never smoked" , "Unknown", "formerly smoked", "smokes"], [0, 1, 2, 3])

        # Gender'ları label'la 
        self.df["gender"] = self.df["gender"].replace(["Male", "Female", "Other"], [0, 1, 2])

        # Evlenip Evlenmediklerini Label'la 
        self.df["ever_married"] = self.df["ever_married"].replace(["No", "Yes"], (0, 1))

        # İş tiplerini label'la 
        self.df["work_type"] = self.df["work_type"].replace(["Never_worked", "children", "Self-employed", "Private", "Govt_job"], [0, 1, 2, 3, 4])

        # Residence_Type'ı Label'la 
        self.df["Residence_type"] = self.df["Residence_type"].replace(["Urban", "Rural"], [0, 1])


    def set_datasets(self): 
        # Train - Val - Test Split = (70, 15, 15)
        
        # First Train Split 
        self.train_df, temp_df = train_test_split(self.df, train_size=0.7, random_state= self.seed)
        
        # Val-Test split from the temp_df 
        self.val_df, self.test_df = train_test_split(temp_df, test_size=0.5, random_state= self.seed)
        
    