import io
import json

import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vit import VisionTransformer


app = Flask(__name__)


# Loading the model 
model = VisionTransformer()
model.load_state_dict(torch.load("./checkpoints/model.pt"))
model.eval()

# Predict API Route 
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # TODO: fix from request body to tensor for prediction 
        digit_tensor = request.body
        predicted_digit, probability_dict = model.predict(digit_tensor)
        return jsonify({'predicted_digit': predicted_digit, 'probability_dict': probability_dict})


if __name__ == '__main__':
    app.run(port=5000, debug=True)