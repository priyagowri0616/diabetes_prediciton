from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form input values
        input_features = [float(x) for x in request.form.values()]
        
        # Convert to numpy array and reshape
        input_array = np.array(input_features).reshape(1, -1)
        
        # Scale input data
        input_scaled = scaler.transform(input_array)
        
        # Predict using model
        prediction = model.predict(input_scaled)
        
        # Interpret prediction
        result = "🩸 Diabetic" if prediction[0] == 1 else "✅ Not Diabetic"
        
        return render_template('index.html', prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)