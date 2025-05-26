"""
app.py

Flask API to serve a trained image classification model.

Exposes a `/predict` endpoint that accepts an image via POST and returns
main and subcategory predictions with probabilities and logits.

Used in combination with batch_processing.py to classify refund items.
"""

from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from train_model import CustomResNet18
import io
import os

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model = CustomResNet18(num_main_categories=3, num_sub_categories=6)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load model weights from relative path
model_path = os.path.join("model", "model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
# Set model to evaluation mode
model.eval()

# Define preprocessing transformations for incoming images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define label mappings
combined_mapping = {
    0: ("bottomwear", "pants"),
    1: ("bottomwear", "shorts"),
    2: ("footwear", "heels"),
    3: ("footwear", "sneakers"),
    4: ("upperwear", "jacket"),
    5: ("upperwear", "shirt"),
}

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests to the /predict endpoint to classify an uploaded image.

    Expects an image file in the request. Applies preprocessing, runs inference using
    the trained model, and returns predicted main and subcategory labels along with 
    probabilities and logits.

    Returns:
        JSON response containing:
            - predicted_main_category
            - predicted_main_category_index
            - main_category_probability
            - predicted_subcategory
            - predicted_subcategory_index
            - subcategory_probability
            - main_category_logits
            - sub_category_logits
    """
    
    if 'file' not in request.files:
        return jsonify({"error": "no file uploaded"}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "no file selected"}), 400
    
    try:
        # Transform image
        img = Image.open(io.BytesIO(file.read()))
        img = transform(img).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            main_output, sub_output = model(img)
            
            main_probabilities = torch.nn.functional.softmax(main_output, dim=1)
            sub_probabilities = torch.nn.functional.softmax(sub_output, dim=1)

            main_predicted_idx = torch.argmax(main_probabilities, dim=1).item()
            sub_predicted_idx = torch.argmax(sub_probabilities, dim=1).item()

            if sub_predicted_idx not in combined_mapping:
                return jsonify({"error": f"Unknown subcategory index: {sub_predicted_idx}"}), 500
            
            main_category, sub_category = combined_mapping[sub_predicted_idx]
            main_probability = main_probabilities[0][main_predicted_idx].item()
            sub_probability = sub_probabilities[0][sub_predicted_idx].item()

        return jsonify({
            "predicted_main_category": main_category,
            "predicted_main_category_index": main_predicted_idx,
            "main_category_probability": main_probability,
            "predicted_subcategory": sub_category,
            "predicted_subcategory_index": sub_predicted_idx,
            "subcategory_probability": sub_probability,
            "main_category_logits": main_output[0].tolist(),
            "sub_category_logits": sub_output[0].tolist()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)