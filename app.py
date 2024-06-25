from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the trained model and scalers
model = load_model('trained_model.h5')
preprocessor = joblib.load('preprocessor.pkl')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

def predict_likes(hashtag, caption, location):
    # Create a DataFrame for the new data
    new_data = pd.DataFrame({
        'Hashtag': [hashtag],
        'Caption': [caption],
        'Location': [location]
    })

    # Apply the preprocessing
    X_new = preprocessor.transform(new_data)
    X_new = scaler_X.transform(X_new)

    # Predict using the trained model
    y_pred_normalized = model.predict(X_new)
    
    # Denormalize the predictions
    y_pred = scaler_y.inverse_transform(y_pred_normalized).flatten()
    
    return int(y_pred[0])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    hashtag = data['hashtag']
    caption = data['caption']
    location = data['location']
    
    predicted_likes = predict_likes(hashtag, caption, location)
    
    return jsonify({'predicted_likes': predicted_likes})

if __name__ == "__main__":
    app.run(debug=True)
