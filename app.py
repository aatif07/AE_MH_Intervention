import numpy as np
import pandas
from flask import Flask, request, jsonify, render_template, request
import pickle
from fileinput import filename


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

    # Return HTML snippet that will render the table
    return data.to_html()


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

