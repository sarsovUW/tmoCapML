import pickle
import numpy as np
from flask import Flask, request, jsonify
import json
import sklearn

app = Flask(__name__)

# Load the machine learning model from the pickle file
with open('model2.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json(force=True)
    
    data_string = json.dumps(data)
    # Make a prediction using the model
    prediction = model.predict(np.array(json.loads(data_string)).reshape(-1, 5))

    # Return the prediction as JSON
    return jsonify(prediction.tolist())

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
