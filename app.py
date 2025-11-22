import flask
# 1. ADDED render_template HERE
from flask import Flask, request, jsonify, render_template 
import joblib
import pandas as pd
import numpy as np
import json

# Initialize the Flask app
app = Flask(__name__)

# --- Load All Model Assets ---
try:
    model = joblib.load('random_forest_model.joblib')
    scaler = joblib.load('min_max_scaler.joblib')
    le_bike_name = joblib.load('bike_name_encoder.joblib')
    le_brand = joblib.load('brand_encoder.joblib')
    le_city = joblib.load('city_encoder.joblib')
    le_owner = joblib.load('owner_encoder.joblib')
    with open('outlier_bounds.json', 'r') as f:
        outlier_bounds = json.load(f)
    model_columns = scaler.feature_names_in_
    print("--- Model and preprocessing assets loaded successfully ---")
except FileNotFoundError as e:
    print(f"Error loading model assets: {e}")
    model = None

# --- Define the Preprocessing Function ---
def preprocess_input(input_df_raw):
    input_df = input_df_raw.copy()
    for col in ['bike_name', 'brand', 'city', 'owner']:
        if col in input_df.columns:
            input_df[col] = input_df[col].str.strip().str.lower()
    try:
        input_df['bike_name_encoded'] = le_bike_name.transform(input_df['bike_name'])
        input_df['brand_encoded'] = le_brand.transform(input_df['brand'])
        input_df['city_encoded'] = le_city.transform(input_df['city'])
        input_df['owner_encoded'] = le_owner.transform(input_df['owner'])
    except ValueError as e:
        return None, f"Error encoding data: {e}. An input category may be unknown to the model."

    current_year = pd.Timestamp('now').year
    input_df['bike_age'] = current_year - input_df['age']
    input_df['km_per_year'] = input_df['kms_driven'] / (input_df['age'] + 1e-6)
    input_df = input_df.drop(['bike_name', 'brand', 'city', 'owner'], axis=1)

    for col in model_columns:
        if col in input_df.columns:
            lower = outlier_bounds['lower'].get(col)
            upper = outlier_bounds['upper'].get(col)
            if lower is not None and upper is not None:
                input_df[col] = np.clip(input_df[col], lower, upper)
                
    input_df[model_columns] = scaler.transform(input_df[model_columns])
    processed_df = input_df[model_columns]
    return processed_df, None

# --- Define the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded. Check server logs.'}), 500
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        processed_df, error = preprocess_input(input_df)
        if error:
            return jsonify({'error': error}), 400
        prediction = model.predict(processed_df)
        return jsonify({'predicted_price': round(prediction[0], 2)})
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

# 2. ADDED THIS NEW 'home' ROUTE
# --- Define the '/' (HOME) ENDPOINT ---
@app.route('/')
def home():
    # Flask will look inside the "templates" folder for this file
    return render_template('index.html')

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)