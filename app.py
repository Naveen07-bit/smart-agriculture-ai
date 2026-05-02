from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import os

app = Flask(__name__)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))

API_KEY = "6bb9d179207b8e62a37b85e7c4eba388"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    location = request.form["location"]

    # 🌦 Weather API
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
        response = requests.get(url).json()
        temp = response["main"]["temp"]
        humidity = response["main"]["humidity"]
    except:
        temp = 25
        humidity = 60

    # Inputs
    n = float(request.form["n"])
    p = float(request.form["p"])
    k = float(request.form["k"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    data = [n, p, k, temp, humidity, ph, rainfall]

    probs = model.predict_proba([data])[0]
    classes = model.classes_

    prediction = classes[np.argmax(probs)]
    confidence = round(max(probs) * 100, 2)

    top_indices = probs.argsort()[-3:][::-1]
    top_crops = [(classes[i], round(probs[i]*100,2)) for i in top_indices]

    profit_map = {
        "rice": 30000,
        "wheat": 25000,
        "maize": 20000,
        "cotton": 40000,
        "sugarcane": 45000
    }

    profit = profit_map.get(prediction, "N/A")

    return render_template("index.html",
                           result=prediction,
                           confidence=confidence,
                           location=location,
                           temp=temp,
                           humidity=humidity,
                           profit=profit,
                           top_crops=top_crops,
                           probs=[round(float(p)*100,2) for p in probs],
                           labels=list(classes))

if __name__ == "__main__":
    app.run(debug=True)