from flask import Flask, request, jsonify, render_template

import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("mental_health_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = [
        data["Gender"],
        data["Age"],
        data["Course"],
        data["Currenrt Year"],
        data["CGPA"],
        data["Marital status"]
    ]

    prediction = model.predict([features])[0]

    risk_map = {
        0: "Low Risk",
        1: "Moderate Risk",
        2: "High Risk"
    }

    return jsonify({
        "prediction": risk_map[int(prediction)]
    })

if __name__ == "__main__":
    app.run(debug=True)