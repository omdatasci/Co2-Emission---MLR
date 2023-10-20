import pandas as pd
import numpy as np
import pickle

from flask import Flask, render_template, request
app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_dt = {
        "ENGINESIZE" : float(request.form['Engine size']),
        "CYLINDERS" : float(request.form['Cylinders']),
        "FUELCONSUMPTION_COMB" : float(request.form['Fuel consumption combination'])
    }

    input_df = pd.DataFrame([input_dt])
    prediction = model.predict(input_df)[0]

    return render_template('result.html', predictions=prediction)


if __name__=='__main__':
    app.run(debug=True)



