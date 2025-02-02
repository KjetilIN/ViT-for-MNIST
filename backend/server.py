import io
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.vit import VisionTransformer

app = Flask(__name__)
CORS(app)

# Loading the model
model = VisionTransformer()
model.load_state_dict(torch.load("./checkpoints/model.pt"))
model.eval()

def debug_tensor(tensor, name="tensor"):
    """Print debug information about a tensor"""
    print(f"\n=== {name} Debug Info ===")
    print(f"Shape: {tensor.shape}")
    print(f"Data type: {tensor.dtype}")
    print(f"Min value: {tensor.min()}")
    print(f"Max value: {tensor.max()}")
    print(f"Mean value: {tensor.mean()}")
    print(f"Sample values:\n{tensor[0, :5, :5]}\n")  # Print 5x5 sample

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the matrix from request body
            data = request.get_json()
            if 'matrix' not in data:
                return jsonify({'error': 'No matrix found in request body'}), 400
            
            # Print raw input data
            print("\n=== Raw Input Data ===")
            print(f"Length of input array: {len(data['matrix'])}")
            print(f"Sample values: {data['matrix'][:10]}")
            
            # Convert the 1D array to a 28x28 numpy array
            matrix = np.array(data['matrix'], dtype=np.float32)
            matrix = matrix.reshape(28, 28)
            
            # Normalize the data
            # If input is 0-255, normalize to 0-1
            if matrix.max() > 1.0:
                matrix = matrix / 255.0
            
            # Convert to tensor and add batch dimension
            digit_tensor = torch.from_numpy(matrix).unsqueeze(0)
            
            # Debug prints
            debug_tensor(digit_tensor, "Input Tensor")
            
            # Make prediction
            with torch.no_grad():
                predicted_digit, probability_dict = model.predict(digit_tensor)
            
            # Debug prints for prediction
            print("\n=== Prediction Debug Info ===")
            print(f"Predicted digit: {predicted_digit}")
            print("Probabilities:", {k: float(v) for k, v in probability_dict.items()})
            
            # Convert any numpy/torch types to Python native types
            probability_dict = {int(k): float(v) for k, v in probability_dict.items()}
            
            return jsonify({
                'prediction': int(predicted_digit),
                'probabilities': probability_dict
            })
            
        except Exception as e:
            print(f"\n=== Error ===\n{str(e)}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)