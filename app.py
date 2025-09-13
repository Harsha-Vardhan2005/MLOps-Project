import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from mlProject import logger

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")
    return "Training Successful!"

@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            gender = request.form['gender']
            seniorcitizen = int(request.form['seniorcitizen'])
            partner = request.form['partner']
            dependents = request.form['dependents']
            tenure = int(request.form['tenure'])
            phoneservice = request.form['phoneservice']
            multiplelines = request.form['multiplelines']
            internetservice = request.form['internetservice']
            onlinesecurity = request.form['onlinesecurity']
            onlinebackup = request.form['onlinebackup']
            deviceprotection = request.form['deviceprotection']
            techsupport = request.form['techsupport']
            streamingtv = request.form['streamingtv']
            streamingmovies = request.form['streamingmovies']
            contract = request.form['contract']
            paperlessbilling = request.form['paperlessbilling']
            paymentmethod = request.form['paymentmethod']
            monthlycharges = float(request.form['monthlycharges'])
            totalcharges = float(request.form['totalcharges'])

            data = [gender, seniorcitizen, partner, dependents, tenure, phoneservice,
                   multiplelines, internetservice, onlinesecurity, onlinebackup,
                   deviceprotection, techsupport, streamingtv, streamingmovies,
                   contract, paperlessbilling, paymentmethod, monthlycharges, totalcharges]
            
            data = np.array(data).reshape(1, 19)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)