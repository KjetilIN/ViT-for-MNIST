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
model.load_state_dict(torch.load("./checkpoints/model.pt", map_location=torch.device("cpu")))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the matrix from request body
            data = request.get_json()
            if 'matrix' not in data:
                return jsonify({'error': 'No matrix found in request body'}), 400
            
            # Convert to numpy and reshape
            matrix = np.array(data['matrix'], dtype=np.float32).reshape(28, 28)
            
            # Convert to tensor
            digit_tensor = torch.from_numpy(matrix).unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            with torch.no_grad():
                predicted_digit, probability_dict = model.predict(digit_tensor)
                for k, v in probability_dict.items():
                    print(f"Raw pred: Digit {k}: {v:.4f}")
            
            # Round probabilities to 4 decimal places and convert to Python types
            rounded_probs = {int(k): round(float(v), 4) for k, v in probability_dict.items()}
            
            # Debug print
            print("\nPrediction Details:")
            sorted_probs = sorted(rounded_probs.items(), key=lambda x: x[1], reverse=True)
            for digit, prob in sorted_probs:
                print(f"Digit {digit}: {prob:.4f}")
            
            return jsonify({
                'prediction': int(predicted_digit),
                'probabilities': rounded_probs
            })
            
        except Exception as e:
            print(f"\n=== Error ===\n{str(e)}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)