from flask import Flask, request, render_template
import pickle
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

dtr = pickle.load(open("dtr.pkl", "rb"))
preprocesser = pickle.load(open("preprocesser.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    Area = request.form["area"]
    Item = request.form["item"]
    Year = int(request.form["year"])
    average_rain_fall_mm_per_year = float(request.form["rainfall"])
    pesticides_tonnes = float(request.form["pesticides"])
    avg_temp = float(request.form["temperature"])

    features = pd.DataFrame([{
        'Area': Area,
        'Item': Item,
        'Year': Year,
        'average_rain_fall_mm_per_year': average_rain_fall_mm_per_year,
        'pesticides_tonnes': pesticides_tonnes,
        'avg_temp': avg_temp
    }])

    transform_features = preprocesser.transform(features)
    predicted_yield = dtr.predict(transform_features).reshape(-1, 1)
    return f"<h2 style='text-align:center;'>Predicted yield (hg/ha): {predicted_yield[0][0]:.2f}</h2>"

if __name__ == "__main__":
    app.run(debug=True)
