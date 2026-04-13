from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model & encoder
try:
    model = pickle.load(open(r"D:\Crop_predction\xgb_model.pkl", "rb"))
    le = pickle.load(open(r"D:\Crop_predction\label_encoder.pkl", "rb"))
    print("✅ Model & Encoder loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        ndvi = float(request.form['ndvi'])
        temp = float(request.form['temp'])
        humidity = float(request.form['humidity'])
        ndwi = float(request.form['ndwi'])
        ndmi = float(request.form['ndmi'])

        print("Inputs:", ndvi, temp, humidity, ndwi, ndmi)

        # Arrange features (IMPORTANT: same order as training)
        features = np.array([[ndvi, temp, humidity, ndwi, ndmi]])

        # Prediction
        pred = model.predict(features)
        print("Raw Prediction:", pred)

        # Decode label
        result = le.inverse_transform(pred)[0]

        print("Final Result:", result)

        return render_template("result.html", prediction=result)

    except Exception as e:
        print("❌ Error:", e)
        return f"⚠️ Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)