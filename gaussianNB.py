# -*- coding: utf-8 -*-

#from google.colab import files
#upload = files.upload()

# Importing libraries
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# %matplotlib inline

# Reading the train.csv by removing the last column since it's an empty column
DATA_PATH = "dataset-gaussian_nb/Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis = 1)

# Checking whether the dataset is balanced or not
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})

#plt.figure(figsize = (18,8))
#sns.barplot(x = "Disease", y = "Counts", data = temp_df)
#plt.xticks(rotation=90)
#plt.show()

# Encoding the target value into numerical value using LabelEncoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test =train_test_split(
X, y, test_size = 0.2, random_state = 24)

#print(f"Train: {X_train.shape}, {y_train.shape}")
#print(f"Test: {X_test.shape}, {y_test.shape}")

#from google.colab import files
#upload = files.upload()

final_nb_model = GaussianNB()
final_nb_model.fit(X, y)
# Reading the test data
test_data = pd.read_csv("dataset-gaussian_nb/Testing.csv").dropna(axis=1)
 
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])
nb_preds = final_nb_model.predict(test_X)

symptoms = X.columns.values

# Creating a symptom index dictionary to encode the input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_
}

def diseasePrecaution(disease) :
    disease_dtls = {}
    
    desc_df = pd.read_csv("symptoms-dataset/symptom_Description.csv")
    precaution_df = pd.read_csv("symptoms-dataset/symptom_precaution.csv")
    
    desc = desc_df[desc_df["Disease"] == disease.title()]
    desc = [*desc["Description"].values]
    
    prec = precaution_df[precaution_df["Disease"] == disease.title()]
    prec.drop(columns = "Disease", inplace=True)
    
    prec_lst = [*prec["Precaution_1"].values, *prec["Precaution_2"].values, *prec["Precaution_3"].values, *prec["Precaution_4"].values]
    
    disease_dtls = {disease:{"desc" : desc, "precaution" : prec}}
#    print(desc)
#    print(prec_lst)
    
    return disease_dtls
    
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    symptoms = [n.strip().title() for n in symptoms]
    
    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        if symptom in data_dict["symptom_index"].keys() :
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
        
    # reshaping the input data and converting it into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
    
    # generating individual outputs
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    
    return diseasePrecaution(nb_prediction)


#print(predictDisease("Depression, irritability"))
#print(predictDisease("Shivering,Chills"))
#print(predictDisease("Neck Pain,Dizziness"))
#print(predictDisease("Chest Pain,Dizziness,Headache"))
