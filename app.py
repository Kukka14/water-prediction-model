from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

app = Flask(__name__)  # Correctly initializes the Flask application

# Load the dataset and preprocess it
dataset = pd.read_csv('balanced_dataset.csv')  # Ensure the correct path to your dataset

# Train the model
X = dataset.drop('Target', axis=1)  # Features
y = dataset['Target']  # Target variable

# Resample the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def get_started():
    return render_template('get_started.html')  # Render the "Get Started" page

@app.route('/form')
def form():
    return render_template('index.html')  # Render the index.html when the user clicks "Get Started"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()  # Get form data
    input_data = pd.DataFrame([data])  # Create a DataFrame from the input data
    
    # Ensure the input data types match the model's expectations
    input_data = input_data.astype(float)  # Convert to float
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Convert prediction to a standard Python type
    prediction_value = int(prediction[0])  # Convert to int for JSON serialization
    
    return jsonify({'prediction': prediction_value})  # Return the prediction as JSON

if __name__ == '__main__':  # Check if the script is run directly
    app.run(debug=False,host='0.0.0.0')  # Start the Flask application in debug mode
