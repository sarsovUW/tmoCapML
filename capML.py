import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the machine learning model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json(force=True)

    # Make a prediction using the model
    prediction = model.predict(data)

    # Return the prediction as JSON
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run()
