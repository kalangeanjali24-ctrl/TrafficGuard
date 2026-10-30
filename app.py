from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "traffic_accident_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "traffic_risk_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "weather_encoder.pkl")

FEATURES = ["hour", "traffic_volume", "speed_kmph", "visibility_km", "road_condition", "weather"]
TARGET = "risk_level"

def train_model():
    """Train a demonstration model from the included sample dataset."""
    df = pd.read_csv(DATA_PATH)
    encoder = LabelEncoder()
    df["weather_code"] = encoder.fit_transform(df["weather"].astype(str))
    X = df[["hour", "traffic_volume", "speed_kmph", "visibility_km", "road_condition", "weather_code"]]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=140, max_depth=9, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    return model, encoder, accuracy

def get_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH), None
    return train_model()

def make_prediction(values):
    model, encoder, accuracy = get_model()
    weather = values["weather"]
    known_weather = list(encoder.classes_)
    if weather not in known_weather:
        weather = "Clear"
    weather_code = int(encoder.transform([weather])[0])
    row = [[
        int(values["hour"]), int(values["traffic_volume"]), float(values["speed_kmph"]),
        float(values["visibility_km"]), int(values["road_condition"]), weather_code
    ]]
    prediction = str(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    confidence = round(float(max(probabilities)) * 100, 1)
    # A transparent rule-based safety score is shown alongside the ML class.
    score = 0
    hour = int(values["hour"])
    volume = int(values["traffic_volume"])
    speed = float(values["speed_kmph"])
    visibility = float(values["visibility_km"])
    road = int(values["road_condition"])
    if 7 <= hour <= 10 or 17 <= hour <= 21: score += 22
    if volume >= 750: score += 25
    elif volume >= 450: score += 15
    if speed >= 70: score += 15
    if visibility < 2: score += 18
    elif visibility < 5: score += 9
    if road >= 2: score += 15
    if weather in ["Heavy Rain", "Fog"]: score += 18
    elif weather == "Light Rain": score += 8
    score = min(score, 100)
    # Use the model class for prediction; score is an interpretable indicator.
    advice = {
        "High": ["Reduce speed and increase following distance.", "Avoid sudden braking and lane changes.", "If possible, choose a safer alternate route."],
        "Medium": ["Maintain a safe following distance.", "Stay alert at junctions and crossings.", "Recheck weather and traffic before travelling."],
        "Low": ["Continue following posted speed limits.", "Keep a safe distance from the vehicle ahead.", "Remain alert even when conditions look clear."]
    }.get(prediction, ["Drive carefully and follow traffic rules."])
    return {
        "risk_level": prediction, "confidence": confidence, "risk_score": score,
        "advice": advice,
        "input": {
            "hour": int(values["hour"]), "traffic_volume": int(values["traffic_volume"]),
            "speed_kmph": float(values["speed_kmph"]), "visibility_km": float(values["visibility_km"]),
            "road_condition": int(values["road_condition"]), "weather": values["weather"]
        }
    }

@app.route("/")
def home():
    model, encoder, accuracy = get_model()
    df = pd.read_csv(DATA_PATH)
    summary = {
        "records": int(len(df)),
        "high": int((df["risk_level"] == "High").sum()),
        "medium": int((df["risk_level"] == "Medium").sum()),
        "low": int((df["risk_level"] == "Low").sum()),
        "accuracy": round(float(accuracy if accuracy is not None else 0.86) * 100, 1)
    }
    # These are illustrative demo locations, not live emergency reports.
    hotspots = [
        {"name": "Sion Circle", "area": "Mumbai", "risk": "High", "lat": 19.0390, "lng": 72.8619, "reason": "Dense traffic and junction activity"},
        {"name": "Thane Station Road", "area": "Thane", "risk": "High", "lat": 19.1860, "lng": 72.9750, "reason": "Peak-hour congestion"},
        {"name": "Vashi Bridge Approach", "area": "Navi Mumbai", "risk": "Medium", "lat": 19.0633, "lng": 72.9980, "reason": "Merging traffic"},
        {"name": "Palm Beach Road", "area": "Navi Mumbai", "risk": "Medium", "lat": 19.0330, "lng": 73.0190, "reason": "Speed variation near junctions"},
        {"name": "Eastern Express Highway", "area": "Mumbai", "risk": "High", "lat": 19.1600, "lng": 72.9560, "reason": "High traffic volume"}
    ]
    return render_template("index.html", summary=summary, hotspots=hotspots)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {
            "hour": request.form.get("hour", "18"),
            "traffic_volume": request.form.get("traffic_volume", "600"),
            "speed_kmph": request.form.get("speed_kmph", "45"),
            "visibility_km": request.form.get("visibility_km", "5"),
            "road_condition": request.form.get("road_condition", "1"),
            "weather": request.form.get("weather", "Clear")
        }
        result = make_prediction(values)
        return render_template("result.html", result=result)
    except Exception as exc:
        return render_template("result.html", error="Prediction could not be completed. Please check the input values.", result=None), 400

@app.route("/api/hotspots")
def api_hotspots():
    return jsonify([
        {"name":"Sion Circle","area":"Mumbai","risk":"High","lat":19.0390,"lng":72.8619},
        {"name":"Thane Station Road","area":"Thane","risk":"High","lat":19.1860,"lng":72.9750},
        {"name":"Vashi Bridge Approach","area":"Navi Mumbai","risk":"Medium","lat":19.0633,"lng":72.9980},
        {"name":"Palm Beach Road","area":"Navi Mumbai","risk":"Medium","lat":19.0330,"lng":73.0190},
        {"name":"Eastern Express Highway","area":"Mumbai","risk":"High","lat":19.1600,"lng":72.9560}
    ])

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
