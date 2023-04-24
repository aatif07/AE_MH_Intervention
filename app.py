import os

import numpy as np
import pandas
from flask import Flask, request, jsonify, render_template, request
import pickle
from fileinput import filename
import csv
from pathlib import Path
import os




# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

# Root endpoint
@app.get('/')
def upload():
    return render_template('index.html')

@app.post('/view')
def view():
    # Read the File using Flask request
    file = request.files['file']
    # save file in local directory
    file.save(file.filename)

    # Parse the data as a Pandas DataFrame type
    data = pandas.read_excel(file)
    prediction = model.predict(data)
    abc = (prediction==1).sum()
    data['Prediction'] = model.predict(data)
    cba = data.to_excel('mhpat.xlsx')
    dfold = str(os.path.join(Path.home(), "Downloads"))
    data.to_csv(os.path.join(dfold,'mh11.xlsx'))

    # Return HTML snippet that will render the table
    return render_template("index.html", prediction_text = "Out of the list of patients, {} will have MH interventions in the near future".format(abc)), cba, data.to_excel('mhpat1.xlsx')


@app.route("/apcmhref.html")
def ApcMHRef():
    return render_template("apcmhref.html")

@app.route("/apcprimdiag.html")
def Apcprimdiag():
    return render_template("apcprimdiag.html")

@app.route("/apcprimsubcat.html")
def Apcprimsubcat():
    return render_template("apcprimsubcat.html")

@app.route("/apcsecdiag.html")
def Apcsecdiag():
    return render_template("apcsecdiag.html")

@app.route("/apcsecsubcat.html")
def Apcsecsubcat():
    return render_template("apcsecsubcat.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [int(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if prediction==0:
        prediction="This patient will not have MH referral"
    else:
        prediction="This patient will have MH referral"
    return render_template("aemhref.html", prediction_text = "{}".format(prediction))

@app.route("/index.html")
def HomePage():
    return render_template("index.html")

@app.route("/aemhref.html", methods = ["POST"])
def AeMHRef():
    float_features = [int(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if prediction == 0:
        prediction = "This patient will not have MH referral"
    else:
        prediction = "This patient will have MH referral"
    return render_template("aemhref.html", prediction_text="{}".format(prediction))

@app.route("/aediag.html")
def AeDiag():
    return render_template("aediag.html")

@app.route("/aesubcat.html")
def AeSubCat():
    return render_template("aesubcat.html")

if __name__ == "__main__":
    app.run(debug=True)

