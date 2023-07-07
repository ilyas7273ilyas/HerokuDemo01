import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

heart_data = pd.read_csv('heart_new.csv')  # Kaggle Dataset
heart_data = heart_data.drop(['Race'], axis=1)

heart_data['HeartDisease'].replace({'Yes':1,'No':0},inplace=True)
heart_data['Smoking'].replace({'Yes':3,'No':2},inplace=True)
heart_data['AlcoholDrinking'].replace({'Yes':3,'No':2},inplace=True)
heart_data['Stroke'].replace({'Yes':3,'No':2},inplace=True)
heart_data['DiffWalking'].replace({'Yes':3,'No':2},inplace=True)
heart_data['Sex'].replace({'Male':3,'Female':2},inplace=True)
heart_data['Diabetic'].replace({'Yes':3,'No':2,'No, borderline diabetes':4,'Yes (during pregnancy)':5},inplace=True)
heart_data['PhysicalActivity'].replace({'Yes':3,'No':2},inplace=True)
heart_data['Asthma'].replace({'Yes':3,'No':2},inplace=True)
heart_data['AgeCategory'].replace({'18-24':2,'25-29':3,'30-34':4,'35-39':5,'40-44':6,'45-49':7,'50-54':8,'55-59':9,'60-64':10,'65-69':11,'70-74':12,'75-79':13,'80 or older':14},inplace=True)
heart_data['GenHealth'].replace({'Very good':2,'Fair':3,'Good':4,'Poor':5,'Excellent':6},inplace=True)

X = heart_data.drop(columns='HeartDisease', axis=1)
Y = heart_data['HeartDisease']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=3)

# Logistic Regression
model = LogisticRegression(max_iter=1025)

# Training the logistic regression model with training data
model.fit(X_train, Y_train)

# accuracy on training data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on testing data: ', testing_data_accuracy)

# Creating a pickle file for the classifier
filename = 'heart-disease-prediction-Logistic-Regression-model.pkl'
pickle.dump(model, open(filename, 'wb'))