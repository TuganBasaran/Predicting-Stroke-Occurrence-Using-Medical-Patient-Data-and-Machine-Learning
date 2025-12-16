from data_factory.data_factory import StrokeDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

path_file = "data/data.csv"


# Dataset Hazırlandı
dataset = StrokeDataset(path_file, 42) # Parameters: Path_File, seed 

print('-' * 50)
print(dataset.train_df.columns) 
print(dataset.train_df.values[0])


# Bu tarz hiperparametreler oynanabilir - şu anki parametreler arbitrary'dir 
model = DecisionTreeClassifier(criterion= "entropy", splitter= "best") 



model.fit(dataset.X_train, dataset.Y_train)

preds = model.predict(dataset.X_test)


Y_test = dataset.Y_test

report = classification_report(Y_test, preds)

conf_matrix = confusion_matrix(Y_test, preds)

print('-' * 50)
print(report)
print('-' * 50)

print('-' * 50)
print(conf_matrix)
print('-' * 50)