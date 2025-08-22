from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
import pickle

from joblib import load

# Load trained model
model = load("iris_model.joblib")

# Load encoder
encoder = pickle.load(open("encoder.pkl", "rb"))

app = Flask(__name__)

# Home page (form)
@app.route('/')
def index():
    return render_template('home.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form and convert to float
        data1 = float(request.form['a'])
        data2 = float(request.form['b'])
        data3 = float(request.form['c'])
        data4 = float(request.form['d'])
        
        # Prepare input array
        arr = pd.DataFrame([[data1, data2, data3, data4]],
                       columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        
        # Make prediction
        pred = model.predict(arr)
        species = encoder.inverse_transform(pred)[0]
        
        return render_template('after.html', data=pred[0])
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
