import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.io.feather_format import read_feather
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix
from evidently.dashboard import Dashboard
from evidently.tabs import ClassificationPerformanceTab, DataDriftTab

import mlflow
import shap
import sys
from alibi_detect.cd import ChiSquareDrift, KSDrift

def create_mlflow_run():
        print()
        print("MLFlow run started...")
        # get the dataset name
        p_dataset = "data/%s.csv"%(sys.argv[2]) if len(sys.argv) > 2 else 'data.csv'
        p_model = sys.argv[4] if len(sys.argv) > 4 else 'RandomForestClassifier'
        print()
        print("Loading dataset %s ..."%p_dataset)

        # ##### LOAD THE DATA #####
        data = pd.read_csv(p_dataset)
        print() 
        print("Pre-processing & Cleansing...")
        data = data.drop(columns=['Loan_ID']) ## Dropping Loan ID
        categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Credit_History','Loan_Amount_Term']
        numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
        numerical_data = data[numerical_columns].copy()
        categorical_data = data[categorical_columns].copy()
        categorical_data["Gender"] = categorical_data["Gender"].map({"Male":1,"Female":0})
        categorical_data["Married"] = categorical_data["Married"].map({"Yes":1,"No":0})
        categorical_data["Dependents"] = categorical_data["Dependents"].map({"0":0,"1":1,"2":2,"3+":3})
        categorical_data["Education"] = categorical_data["Education"].map({"Graduate":1, "Not Graduate":0})
        categorical_data["Self_Employed"] = categorical_data["Self_Employed"].map({"Yes":1, "No":0})
        categorical_data["Loan_Amount_Term"] = categorical_data["Loan_Amount_Term"].map({12:0,36:1,60:2,84:3,120:4,180:5,240:6,300:7,360:8,480:9})
        categorical_data["Property_Area"] = categorical_data["Property_Area"].map({"Urban":2,"Semiurban":1, "Rural":0})
        categorical_data = categorical_data.to_numpy().astype(int)

        # ##### PLOT CATEGORICAL COLUMNS #####
        sns.set(style="white", context="talk")
        sns.color_palette("rocket")
        fig,axes = plt.subplots(4,2,figsize=(15,25))
        for idx,cat_col in enumerate(categorical_columns):
            row,col = idx//2,idx%2
            sns.countplot(x=cat_col, data=data, hue='Loan_Status', ax=axes[row,col])
        plt.subplots_adjust(hspace=0.5)
        fig.savefig('CATEGORICAL_DATA.png', bbox_inches='tight')

        # ##### PLOT NUMERIC COLUMNS #####
        fig,axes = plt.subplots(1,3,figsize=(17,5))
        for idx,cat_col in enumerate(numerical_columns):
            sns.boxplot(y=cat_col, data=data, x='Loan_Status',ax=axes[idx])
        plt.subplots_adjust(hspace=0.5)
        fig.savefig('NUMERIC_DATA.png', bbox_inches='tight')

        ##### FEATURE ENGINEERING #####
        print()
        print("Feature Engineering...")
        train_df_encoded = pd.get_dummies(data,drop_first=True)
        X = train_df_encoded.drop(columns='Loan_Status_Y')
        X_columns = X.columns
        y = train_df_encoded['Loan_Status_Y']
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25)

        imp = SimpleImputer(strategy='mean')
        imp_train = imp.fit(X_train)
        X_train = imp_train.transform(X_train)
        X_test_tmp = X_test.copy()
        X_test = imp_train.transform(X_test)
        X_test_df = pd.DataFrame(X_test, columns=X_columns)
        imp_numerical_data = imp.fit(numerical_data)
        numerical_data = imp_numerical_data.transform(numerical_data)

        # ##### BUILD ML MODEL #####
        print()
        print("Building the ML Model...")
        model = RandomForestClassifier(n_estimators=50,max_depth=5)
        model.fit(X_train,y_train)
        # print(list(zip(train_df_encoded.columns, model.feature_importances_)))       
        # cross-validation
        xval_scores = cross_val_score(model, X_train, y_train, cv=10)   

        # ##### METRICS #####
        print()
        print("Model evaluation...")
        y_pred = model.predict(X_train)
        train_f1 = f1_score(y_train,y_pred)
        train_acc = accuracy_score(y_train,y_pred)
        y_pred = model.predict(X_test)
        test_f1 = f1_score(y_test,y_pred)
        test_acc = accuracy_score(y_test,y_pred)

        plt.figure(figsize=(30,15))
        plot_confusion_matrix(model, X_test, y_test,
        display_labels=['Approved', 'Rejected'],
        cmap=plt.cm.Blues,
        normalize='pred')
        plt.savefig('CONFUSION_MATRIX.png', bbox_inches='tight')

        # ##### EXPLANATIONS #####
        print()
        print("Generating explanations...")
        plt.figure(figsize=(30,15))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test_tmp, show=False, class_names=['Rejected','Approved'])
        plt.savefig('SHAP_EXPLANATIONS.png', bbox_inches='tight')


        # ##### LOG THE RUN #####
        with mlflow.start_run():
            mlflow.log_param('DATA CATALOG ENTRY', p_dataset)
            mlflow.log_param('MODEL CATALOG ENTRY', p_model)
            mlflow.log_artifact('CATEGORICAL_DATA.png')
            mlflow.log_artifact('NUMERIC_DATA.png')
            mlflow.log_artifact('CONFUSION_MATRIX.png')
            mlflow.log_metric('TRAINING F1 SCORE', train_f1)
            mlflow.log_metric('TESTING F1 SCORE', test_f1)
            mlflow.log_metric('TRAINING ACCURACY', train_acc)
            mlflow.log_metric('TESTING ACCURACY', test_acc)

            print('--'*10)
            print('TRAINING ACCURACY', train_acc)
            print('TESTING ACCURACY', test_acc)
            print('--'*10)
            for score in xval_scores:
                mlflow.log_metric('CROSS VALIDATION SCORE', score)
            mlflow.log_artifact('SHAP_EXPLANATIONS.png')
            print()
            print("MLFlow run completed!")

        # # ##### DRIFT GENERATOR DATASET ######
        print()
        print("Creating dataset with data drift...")
        X_test_df_old = X_test_df.copy()
        for i in range(0,X_test_df.shape[0]):
                ApplicantIncome_Change = np.random.randint(-60,-20)
                LoanAmount_Change = np.random.randint(30,71)
                CoapplicantIncome_Change = np.random.randint(-60,-20)
                X_test_df.loc[i,"ApplicantIncome"] = (1 + (ApplicantIncome_Change/100)) * X_test_df.loc[i,"ApplicantIncome"]
                if(X_test_df.loc[i,"CoapplicantIncome"]!=0):
                        X_test_df.loc[i,"CoapplicantIncome"] = (1 + (CoapplicantIncome_Change/100)) * X_test_df.loc[i,"CoapplicantIncome"]
                X_test_df.loc[i,"LoanAmount"] = (1 + (LoanAmount_Change/100)) * X_test_df.loc[i,"LoanAmount"]
        y_pred_datadrift = model.predict(X_test_df)
        datadrift_test_acc = accuracy_score(y_test,y_pred_datadrift)
        print("Datadrift_Test_Accuracy: ",datadrift_test_acc)
        X_test_df.to_csv("datadrift_inputdata_oldmodel.csv")
        print()
        print("Generating data drift related reports...")
        loan_data_drift_report = Dashboard(tabs=[DataDriftTab])
        loan_data_drift_report.calculate(X_test_df_old, X_test_df, column_mapping=None)
        loan_data_drift_report.save("loan_datadrift_oldone.html")
        reference = pd.DataFrame(X_test, columns=X.columns)
        y_test = y_test.reset_index()
        reference["target"] = y_test["Loan_Status_Y"]
        reference["prediction"] = y_pred
        production = X_test_df.copy()
        production["target"] = y_test["Loan_Status_Y"]
        production["prediction"] = y_pred_datadrift
        loan_datadrift_column_mapping = {}
        loan_datadrift_column_mapping["target"] = "target"
        loan_datadrift_column_mapping["prediction"] = "prediction"
        loan_datadrift_column_mapping["numerical_features"] = numerical_columns
        loan_datadrift_model_performance = Dashboard(tabs=[ClassificationPerformanceTab])
        loan_datadrift_model_performance.calculate(reference, production, column_mapping = loan_datadrift_column_mapping)
        loan_datadrift_model_performance.save("loan_datadrift_classification_performance_oldmodel.html")

        ##### CONCEPT DRIFT DATASET GENERATOR #####
        print()
        print("Creating dataset with Concept Drift...")
        concept_df = pd.read_csv("test.csv")
        concept_df = concept_df.drop(columns=["Loan_ID"])
        concept_df = pd.get_dummies(concept_df,drop_first=True)
        concept_df_columns = concept_df.columns
        imp_concept_df = imp.fit(concept_df)
        concept_df = imp_concept_df.transform(concept_df)
        concept_df = pd.DataFrame(concept_df, columns = concept_df_columns)
        old_y_pred = model.predict(concept_df)
        ground_truth_list = [0] * concept_df.shape[0]
        for i in range(0,concept_df.shape[0]):
                ApplicantIncome_Change = np.random.randint(-20,11)
                LoanAmount_Change = np.random.randint(20,51)
                CoapplicantIncome_Change = np.random.randint(-20,11)
                concept_df.loc[i,"ApplicantIncome"] = (1 + (ApplicantIncome_Change/100)) * concept_df.loc[i,"ApplicantIncome"]
                if(concept_df.loc[i,"CoapplicantIncome"]!=0):
                        concept_df.loc[i,"CoapplicantIncome"] = (1 + (CoapplicantIncome_Change/100)) * concept_df.loc[i,"CoapplicantIncome"]
                concept_df.loc[i,"LoanAmount"] = (1 + (LoanAmount_Change/100)) * concept_df.loc[i,"LoanAmount"]
                if (concept_df.loc[i,"Credit_History"]==1 and (ApplicantIncome_Change or CoapplicantIncome_Change)<0 and LoanAmount_Change > 25):
                        concept_df.loc[i,"Credit_History"] = np.random.randint(0,2)
                ground_truth_list[i] = GroundTruthValue(concept_df.loc[i,"ApplicantIncome"], concept_df.loc[i,"LoanAmount"],ApplicantIncome_Change, LoanAmount_Change)
        new_y_pred = model.predict(concept_df)
        new_acc = accuracy_score(ground_truth_list, new_y_pred)
        print("New accuracy: ",new_acc)
        concept_df["Ground_Truth"] = ground_truth_list
        concept_df.to_csv("concept_drift_input_data.csv", index=False)

        # ##### CONCEPT DRIFT REPORT ######
        print()
        print("Generating Classification Performance Report...")
        reference = production.copy()
        production = concept_df.copy()
        production = production.drop(columns=["Ground_Truth"])
        production["target"] = ground_truth_list
        production["prediction"] = new_y_pred
        loan_column_mapping = {}
        loan_column_mapping["target"] = "target"
        loan_column_mapping["prediction"] = "prediction"
        loan_column_mapping["numerical_features"] = numerical_columns
        loan_model_performance = Dashboard(tabs=[ClassificationPerformanceTab])
        loan_model_performance.calculate(reference, production, column_mapping = loan_column_mapping)
        loan_model_performance.save("loan_classification_performance.html")
        
        ##### NEW INCOMING DATA - DATA DRIFT #####
        print()
        print("New Input Data...")
        inputdata = data.drop(columns=["Loan_Status"])
        column_names = inputdata.columns
        input_dataframe = pd.DataFrame(columns=column_names)
        for i in range(0,10):
            input_dataframe = newinputdata("Loan-1.json", input_dataframe,numerical_data, categorical_data)
            input_dataframe = newinputdata("Loan-2.json", input_dataframe,numerical_data, categorical_data)
            input_dataframe = newinputdata("Loan-1.json", input_dataframe,numerical_data, categorical_data)

def GroundTruthValue(ApplicantIncome, LoanAmount, ApplicantIncome_Change, LoanAmount_Change):
        if(ApplicantIncome>10000 and LoanAmount <=1000):
                return 1
        elif(ApplicantIncome<2500 and LoanAmount >=250):
                return 0
        elif(ApplicantIncome>=7500 and ApplicantIncome_Change > -10 and LoanAmount_Change < 45 and LoanAmount<400):
                return 1
        elif(ApplicantIncome<5000 and ApplicantIncome_Change < -5 and LoanAmount_Change > 35 and LoanAmount > 250):
                return 0
        elif(ApplicantIncome_Change > 0 and LoanAmount_Change < 30 and ApplicantIncome > 5000):
                return 1
        elif(ApplicantIncome_Change < -5 and LoanAmount_Change > 25 and ApplicantIncome < 3500):
                return 0
        else:
                return 1

def newinputdata(filename, input_dataframe, numerical_data, categorical_data):
    f =open(filename,"r")
    data = json.loads(f.read())
    newdataframe = pd.DataFrame.from_dict([data])
    input_dataframe = pd.concat([input_dataframe,newdataframe], axis=0, ignore_index=True)
    if(input_dataframe.shape[0]>=30):
        datadriftdetection(input_dataframe, numerical_data, categorical_data)
    return input_dataframe

def datadriftdetection(input_dataframe, X_ref, categorical_data):
    print()
    print("Checking for Data Drift...")
    print()
    print("Numerical Data...")
    print()
    numeric_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    numerical_input_dataframe = input_dataframe[numeric_columns].copy()
    numerical_input_dataframe_numpy = numerical_input_dataframe.to_numpy()
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Credit_History','Loan_Amount_Term']
    categorical_input_dataframe = input_dataframe[categorical_columns].copy()
    categorical_input_dataframe["Gender"] = categorical_input_dataframe["Gender"].map({"Male":1,"Female":0})
    categorical_input_dataframe["Married"] = categorical_input_dataframe["Married"].map({"Yes":1,"No":0})
    categorical_input_dataframe["Dependents"] = categorical_input_dataframe["Dependents"].map({"0":0,"1":1,"2":2,"3+":3})
    categorical_input_dataframe["Education"] = categorical_input_dataframe["Education"].map({"Graduate":1, "Not Graduate":0})
    categorical_input_dataframe["Self_Employed"] = categorical_input_dataframe["Self_Employed"].map({"Yes":1, "No":0})
    categorical_input_dataframe["Loan_Amount_Term"] = categorical_input_dataframe["Loan_Amount_Term"].map({12:0,36:1,60:2,84:3,120:4,180:5,240:6,300:7,360:8,480:9})
    categorical_input_dataframe["Property_Area"] = categorical_input_dataframe["Property_Area"].map({"Urban":2,"Semiurban":1, "Rural":0})
    categorical_input_dataframe_numpy = categorical_input_dataframe.to_numpy().astype(int)
    cd = KSDrift(X_ref, p_val=.05)
    preds = cd.predict(numerical_input_dataframe_numpy, drift_type='feature', return_p_val=True, return_distance=True)
    fpreds = cd.predict(numerical_input_dataframe_numpy, drift_type='feature')
    print(preds)
    print()
    for f in range(cd.n_features):
        stat = 'K-S'
        fname = numeric_columns[f]
        is_drift = fpreds['data']['is_drift'][f]
        stat_val, p_val = preds['data']['distance'][f], preds['data']['p_val'][f]
        print(f'{fname}-- Drift? {[is_drift]} -- {stat} {stat_val:.3f} -- p-value {p_val:.5f}')   

    print()
    print("Categorical Data")
    print()
    cd = ChiSquareDrift(categorical_data, p_val=.05)
    preds = cd.predict(categorical_input_dataframe_numpy)
    print(preds)
    print()
    print(f"Threshold {preds['data']['threshold']}")
    stat = "Chi2"
    print()
    for f in range(cd.n_features):
        fname = categorical_columns[f]
        is_drift = (preds['data']['p_val'][f] < preds['data']['threshold']).astype(int)
        stat_val, p_val = preds['data']['distance'][f], preds['data']['p_val'][f]
        print(f'{fname} -- Drift? {[is_drift]} -- {stat} {stat_val:.3f} -- p-value {p_val:.3f}')

if __name__ == "__main__":
        create_mlflow_run()



