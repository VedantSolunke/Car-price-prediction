
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load your model and data
df = pd.read_csv('car_data.csv')
inputs = df.drop(['Car_Name', 'Owner', 'Seller_Type'], axis='columns')
target = df.Selling_Price

# Initialize LabelEncoders
fuel_type_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()

# Fit the LabelEncoders on the training data
inputs['Fuel_Type_n'] = fuel_type_encoder.fit_transform(inputs['Fuel_Type'])
inputs['Transmission_n'] = transmission_encoder.fit_transform(inputs['Transmission'])
inputs_n = inputs.drop(['Fuel_Type', 'Transmission', 'Selling_Price'], axis='columns')

model = linear_model.LinearRegression()
model.fit(inputs_n, target)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        year = int(request.form['year'])
        selling_price = float(request.form['selling_price'])
        km_driven = int(request.form['km_driven'])
        fuel_type = fuel_type_encoder.transform([request.form['fuel_type']])[0]
        transmission = transmission_encoder.transform([request.form['transmission']])[0]

        prediction = model.predict([[year, selling_price, km_driven, fuel_type, transmission]])
        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

