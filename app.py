import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("aemhref.html")

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

if __name__ == "__main__":
    app.run(debug=True)

