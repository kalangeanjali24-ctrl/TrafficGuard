# AI-Based Traffic Accident Hotspot Prediction
### TrafficGuard AI — Academic Mini Project

A Flask web application that demonstrates how machine learning can classify road scenarios into **Low**, **Medium**, or **High** accident-risk categories using traffic, weather, visibility, speed, time and road-surface inputs.

## Main features
- Professional responsive dark dashboard UI
- Scenario form for traffic and weather conditions
- Random Forest classification model using scikit-learn
- Risk result page with model confidence and safety recommendations
- Doughnut chart showing the sample dataset's label distribution
- Leaflet map showing **illustrative sample locations** around Mumbai, Thane and Navi Mumbai
- Project methodology and limitations page

## Important academic honesty note
The included CSV is **synthetic sample data created for demonstration**. The map points are also illustrative examples, not verified accident records or live government data. Model confidence is not the actual probability of a crash. Do not use this project for emergency response or real-world route safety decisions.

## Requirements
- Windows 10/11 or another OS with Python 3.10+
- Internet connection for map tiles, Google Fonts and Chart.js CDN
- Python packages listed in `requirements.txt`

## Run on Windows (PowerShell)
1. Extract the ZIP file to a folder such as `Documents\AI_Based_Traffic_Accident_Hotspot_Prediction`.
2. Open that folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   ```
5. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt terminal and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
6. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. Start the website:
   ```powershell
   python app.py
   ```
8. Open the local address printed in the terminal, usually `http://127.0.0.1:5000`.
9. Use **Assess road conditions → Calculate risk** to test a scenario.

The first launch trains the model from the included CSV and saves `traffic_risk_model.pkl` and `weather_encoder.pkl` in the project folder.

## Project structure
```text
AI_Based_Traffic_Accident_Hotspot_Prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── traffic_accident_data.csv
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   └── about.html
└── static/
    ├── css/style.css
    └── js/main.js
```

## How it works
1. **Input:** User selects hour, traffic volume, average speed, visibility, weather and road condition.
2. **Preprocessing:** Weather text is encoded into a numeric value.
3. **Training:** The Random Forest classifier learns from labelled sample scenarios in the CSV.
4. **Prediction:** The trained model predicts a risk class for the submitted scenario.
5. **Output:** The website shows the risk class, the model's highest class probability, the submitted values and safety advice.

## Dataset fields
- `hour`: hour of day from 0 to 23
- `traffic_volume`: estimated vehicles per hour
- `speed_kmph`: average speed in kilometres per hour
- `visibility_km`: visibility distance
- `road_condition`: 0 = dry/good, 1 = normal, 2 = wet/slippery, 3 = damaged/poor
- `weather`: Clear, Cloudy, Light Rain, Heavy Rain or Fog
- `risk_level`: sample label High, Medium or Low

## Viva / external explanation (simple English)
**What is your project?**  
“My project is AI-Based Traffic Accident Hotspot Prediction. It is a web application that demonstrates how traffic and environmental conditions can be used to classify a road scenario into low, medium or high risk.”

**Why did you choose this project?**  
“I chose it because road accidents are an important safety problem. Analysing traffic volume, speed, weather and visibility can help people understand conditions that may require extra care.”

**Where is AI used?**  
“I used a Random Forest machine-learning classifier from scikit-learn. It learns patterns from labelled sample data and predicts a risk category for new input values.”

**What is Random Forest?**  
“Random Forest is a machine-learning algorithm that uses multiple decision trees. The trees make predictions and the model combines their results to select a class.”

**What are the inputs?**  
“Time, traffic volume, average vehicle speed, visibility, weather and road condition.”

**What is the output?**  
“The output is a predicted risk category, a model-confidence value and safety recommendations.”

**What is Flask?**  
“Flask is a Python web framework. It connects the website form with the Python prediction code.”

**What is the role of the dataset?**  
“The CSV file stores example road conditions and their risk labels. The model uses these labelled examples during training.”

**What is the limitation?**  
“The dataset is synthetic and the hotspot markers are illustrative. The prototype is not connected to live accident databases, so it cannot claim to identify verified real-time hotspots.”

**Future scope:**  
“The project could be improved by using verified historical crash records, live traffic feeds, accurate GPS data, weather APIs, more careful model evaluation and expert validation.”

## Troubleshooting
- If `pip` is not recognised, use `python -m pip install -r requirements.txt`.
- If port 5000 is busy, stop the other Flask program or change `app.run(debug=True)` to `app.run(debug=True, port=5001)` and open port 5001.
- If the map is blank, check your internet connection; the prediction form can still work.
- If model training fails, check that `data/traffic_accident_data.csv` exists and has all required columns.
