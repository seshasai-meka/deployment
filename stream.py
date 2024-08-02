from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained KNN model
model = joblib.load("C:/Users/500224610/Downloads/knn_model.pkl")

# If you know the class names, you can specify them here
class_names = ['Setosa', 'Versicolor', 'Virginica','op']

@app.route('/')
def home():
    return "Welcome to the KNN Iris Species Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the request
        data = request.json
        sepal_length = data.get('sepal_length', 5.0)
        sepal_width = data.get('sepal_width', 3.0)
        petal_length = data.get('petal_length', 1.0)
        petal_width = data.get('petal_width', 0.2)
        
        # Prepare input data for the model
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Make prediction
        prediction = model.predict(input_data)
        species = class_names[prediction[0]]
        
        # Return the result as JSON
        return jsonify({'species': species})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
