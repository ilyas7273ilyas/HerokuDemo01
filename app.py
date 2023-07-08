# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Logistic Regression Classifier model
filename = 'heart-disease-prediction-Logistic-Regression-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        BMI = float(request.form['BMI'])
        Smoking = int(request.form.get('Smoking'))
        AlcoholDrinking = int(request.form.get('AlcoholDrinking'))
        Stroke = int(request.form.get('Stroke'))
        PhysicalHealth = int(request.form['PhysicalHealth'])
        MentalHealth = int(request.form['MentalHealth'])
        DiffWalking = int(request.form.get('DiffWalking'))
        Sex = int(request.form.get('Sex'))
        AgeCategory = int(request.form.get('AgeCategory'))
        Diabetic = int(request.form.get('Diabetic'))
        PhysicalActivity = int(request.form.get('PhysicalActivity'))
        GenHealth = int(request.form.get('GenHealth'))
        SleepTime = int(request.form['SleepTime'])
        Asthma = int(request.form.get('Asthma'))

        data = np.array([[BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking,
                          Sex, AgeCategory, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma]], dtype='float64')
        my_prediction = model.predict(data)

        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
