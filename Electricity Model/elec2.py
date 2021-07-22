import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from evidently.dashboard import Dashboard
from evidently.tabs import ClassificationPerformanceTab

dataset = pd.read_csv("elec_data.csv")
dataset = dataset.drop(columns=["id"])
print(dataset.head())
X_first = dataset.drop(columns=["class"])
y_first = dataset["class"]
numerical_columns = X_first.columns
X_train_first = X_first.loc[:4799,:]
X_test_first = X_first.loc[17520:22319,:]
y_train_first = y_first[:4800]
y_test_first = y_first[17520:22320]
model_first = RandomForestClassifier(n_estimators=50, max_depth=3)
model_first.fit(X_train_first,y_train_first)
y_pred_first = model_first.predict(X_test_first)
accuracy_first = metrics.accuracy_score(y_test_first, y_pred_first)
print(accuracy_first)
X_second = dataset.drop(columns=["class"])
y_second = dataset["class"]
X_train_second_first = X_second.loc[:4799,:]
X_train_second_second = X_second.loc[17520:22319,:]
X_train_second = pd.concat([X_train_second_first,X_train_second_second])
X_test_second = X_second.loc[27455:32999,:]
y_train_second_first = y_second[:4800]
y_train_second_second = y_second[17520:22320]
y_train_second = pd.concat([y_train_second_first,y_train_second_second])
y_test_second = y_second[27455:33000]
model_second = RandomForestClassifier(n_estimators=50, max_depth=3)
model_second.fit(X_train_second,y_train_second)
y_pred_second = model_second.predict(X_test_second)
accuracy_second = metrics.accuracy_score(y_test_second, y_pred_second)
print(accuracy_second)
reference = X_test_first.copy()
reference["target"] = y_test_first
reference["prediction"] = y_pred_first
production = X_test_second.copy()
production["target"] = y_test_second
production["prediction"] = y_pred_second
elec_column_mapping = {}
elec_column_mapping["target"] = "target"
elec_column_mapping["prediction"] = "prediction"
elec_column_mapping["numerical_features"] = numerical_columns
elec_model_performance = Dashboard(tabs=[ClassificationPerformanceTab])
elec_model_performance.calculate(reference, production, column_mapping = elec_column_mapping)
elec_model_performance.save("elec_classification_performance.html")