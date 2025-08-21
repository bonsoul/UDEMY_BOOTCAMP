from flask import Flask, render_template, request
import numpy as np
from joblib import load   # use joblib instead of pickle

# Load trained model
model = load("iris_model.joblib")

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    # Get values from form and convert to float
    data1 = float(request.form['a'])
    data2 = float(request.form['b'])
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    
    # Prepare input array
    arr = np.array([[data1, data2, data3, data4]])
    
    # Make prediction
    pred = model.predict(arr)
    
    return render_template('after.html', data=pred[0])

if __name__ == "__main__":
    app.run(debug=True)
