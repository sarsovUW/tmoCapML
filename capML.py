import pickle
import numpy as np
from flask import Flask, request, jsonify
import json
import sklearn

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()
