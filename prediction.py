import os
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

filename = 'Pickle_RL_Model.pkl'
model = joblib.load(filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        age = int(request.form['age'])
        gender = request.form.get('gender')
        chest_pain = request.form.get('chest_pain')
        rest_bps = float(request.form['rest_bps'])
        cholesterol = float(request.form['cholesterol'])
        fasting_blood_sugar = request.form.get('fasting_blood_sugar')
        rest_ecg = request.form.get('rest_ecg')
        thalach = float(request.form['thalach'])
        exercise_angina = request.form.get('exercise_angina')
        old_peak = float(request.form['old_peak'])
        slope = request.form.get('slope')
        ca = request.form.get('ca')
        thalassemia = request.form.get('thalassemia')

        # Prepare input for model
        input_features = [[age,gender, chest_pain, rest_bps, cholesterol, fasting_blood_sugar, rest_ecg, thalach, exercise_angina, old_peak, slope, ca, thalassemia]]

        # Make prediction
        prediction = model.predict(input_features)

        return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
