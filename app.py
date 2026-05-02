from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import requests
import os

app = Flask(__name__)
app.secret_key = "secret123"   # fixed

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

try:
    model = pickle.load(open(model_path, "rb"))
except Exception as e:
    print("Error loading model:", e)
    model = None

# API key
API_KEY = os.environ.get("API_KEY")

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin" and password == "1234":
            session["user"] = username
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ---------------- HOME ----------------
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    if model is None:
        return "Model not loaded properly"

    location = request.form.get("location")

    # Default values
    temp = 25
    humidity = 60

    # Weather API
    if API_KEY and location:
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
            response = requests.get(url).json()

            if "main" in response:
                temp = response["main"]["temp"]
                humidity = response["main"]["humidity"]
        except Exception as e:
            print("Weather API error:", e)

    try:
        n = float(request.form.get("n"))
        p = float(request.form.get("p"))
        k = float(request.form.get("k"))
        ph = float(request.form.get("ph"))
        rainfall = float(request.form.get("rainfall"))

        data = [n, p, k, temp, humidity, ph, rainfall]

        probs = model.predict_proba([data])[0]
        classes = model.classes_

        prediction = classes[np.argmax(probs)]
        confidence = round(max(probs) * 100, 2)

        top_indices = probs.argsort()[-3:][::-1]
        top_crops = [(classes[i], round(probs[i]*100, 2)) for i in top_indices]

        profit_map = {
            "rice": 30000,
            "wheat": 25000,
            "maize": 20000,
            "cotton": 40000,
            "sugarcane": 45000
        }

        profit = profit_map.get(prediction, "N/A")

        return render_template(
            "index.html",
            result=prediction,
            confidence=confidence,
            location=location,
            temp=temp,
            humidity=humidity,
            profit=profit,
            top_crops=top_crops,
            probs=[round(float(p)*100, 2) for p in probs],
            labels=list(classes)
        )

    except Exception as e:
        return f"Error: {str(e)}"


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)