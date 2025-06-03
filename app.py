from flask import Flask
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import requests
import time
import numpy as np
from constants import IMAGENET_CLASSES, IMAGE_URL
import torchvision.models as models

# Initialize Flask App
app = Flask(__name__)

# Define Image Downloading and Preprocessing Function
def load_and_preprocess_image():
    """Downloads an image from IMAGE_URL, preprocesses it, and returns a batch tensor."""
    try:
        img = Image.open(requests.get(IMAGE_URL, stream=True).raw)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None  # Or raise an exception, or handle as appropriate

    img = img.convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    preprocessed_img = preprocess(img)
    batch_img_tensor = preprocessed_img.unsqueeze(0)
    return batch_img_tensor.to('cpu')

# Global Variable for Preprocessed Image
PREPROCESSED_IMAGE_TENSOR = load_and_preprocess_image()

# Define Model Loading Function
def load_model():
    """Loads a pre-trained ResNet model."""
    try:
        model = models.resnet18(pretrained=True)
        model = model.to('cpu')
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Global Variable for Model
MODEL = load_model()

# Warm-up Runs
if MODEL is not None and PREPROCESSED_IMAGE_TENSOR is not None:
    print("Performing warm-up runs...")
    for _ in range(5):  # Perform 5 warm-up runs
        with torch.no_grad():
            MODEL(PREPROCESSED_IMAGE_TENSOR)
    print("Warm-up complete.")
else:
    print("Skipping warm-up runs due to issues with model or image loading.")

# Define Benchmarking Function
def benchmark_inference(model, image_tensor):
    """Performs inference 1000 times and returns latency statistics."""
    latencies = []
    for _ in range(1000):
        start_time = time.time()
        with torch.no_grad():
            model(image_tensor)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
    
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    std_latency = np.std(latencies)
    
    return avg_latency, min_latency, max_latency, std_latency

# Global Variables for Latency Statistics
AVG_LATENCY, MIN_LATENCY, MAX_LATENCY, STD_LATENCY = None, None, None, None

if MODEL is not None and PREPROCESSED_IMAGE_TENSOR is not None:
    print("Performing benchmarking...")
    AVG_LATENCY, MIN_LATENCY, MAX_LATENCY, STD_LATENCY = benchmark_inference(MODEL, PREPROCESSED_IMAGE_TENSOR)
    print(f"Benchmarking complete. Average latency: {AVG_LATENCY:.2f} ms")
else:
    print("Skipping benchmarking due to issues with model or image loading.")

# Define Top 5 Predictions Function
def get_top5_predictions(model, image_tensor, imagenet_classes):
    """Performs inference and returns the top 5 predictions."""
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = F.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    top5_prob_list = top5_prob.tolist()
    top5_catid_list = top5_catid.tolist()
    
    results = []
    for i in range(len(top5_prob_list)):
        results.append({
            'name': imagenet_classes[top5_catid_list[i]],
            'prob': top5_prob_list[i]
        })
    return results

# Global Variable for Top 5 Predictions
TOP_5_PREDICTIONS = []

if MODEL is not None and PREPROCESSED_IMAGE_TENSOR is not None:
    print("Generating top 5 predictions...")
    TOP_5_PREDICTIONS = get_top5_predictions(MODEL, PREPROCESSED_IMAGE_TENSOR, IMAGENET_CLASSES)
    print("Top 5 predictions generated.")
    # For debugging, you might want to print them:
    # for pred in TOP_5_PREDICTIONS:
    #     print(f"- {pred['name']}: {pred['prob']:.4f}")
else:
    print("Skipping generation of top 5 predictions due to issues with model or image loading.")

@app.route('/')
def index():
    # Prepare predictions list for HTML
    predictions_html = ""
    if TOP_5_PREDICTIONS:
        for pred in TOP_5_PREDICTIONS:
            predictions_html += f"<li>Class Name: {pred['name']}, Probability: {pred['prob']*100:.4f}%</li>"
    else:
        predictions_html = "<li>Top 5 predictions not available.</li>"

    # Prepare latency stats for HTML, handling None cases
    avg_latency_html = f"{AVG_LATENCY:.2f} ms" if AVG_LATENCY is not None else "N/A"
    min_latency_html = f"{MIN_LATENCY:.2f} ms" if MIN_LATENCY is not None else "N/A"
    max_latency_html = f"{MAX_LATENCY:.2f} ms" if MAX_LATENCY is not None else "N/A"
    std_latency_html = f"{STD_LATENCY:.2f} ms" if STD_LATENCY is not None else "N/A"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-A">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ImageNet Inference Results</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
            h1, h2 {{ text-align: center; color: #333; }}
            img {{ display: block; margin-left: auto; margin-right: auto; max-width: 90%; height: auto; margin-bottom: 20px; border: 1px solid #ddd; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }}
            .container {{ max-width: 800px; margin: auto; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            .stats, .predictions {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 4px; background-color: #f9f9f9; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin-bottom: 8px; padding: 5px; border-bottom: 1px solid #eee; }}
            li:last-child {{ border-bottom: none; }}
            p {{ line-height: 1.6; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>ImageNet Inference Demo</h1>

            <div class="image-display">
                <h2>Source Image</h2>
                <img src="{IMAGE_URL}" alt="Inference Image">
                <p style="text-align:center;">Image URL used for inference: <a href="{IMAGE_URL}" target="_blank">{IMAGE_URL}</a></p>
            </div>

            <div class="predictions">
                <h2>Top 5 Predictions:</h2>
                <ul>
                    {predictions_html}
                </ul>
            </div>

            <div class="stats">
                <h2>Inference Latency Statistics:</h2>
                <p>Average Latency: {avg_latency_html}</p>
                <p>Minimum Latency: {min_latency_html}</p>
                <p>Maximum Latency: {max_latency_html}</p>
                <p>Standard Deviation: {std_latency_html}</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

if __name__ == '__main__':
    # Check if critical components are loaded before running
    if PREPROCESSED_IMAGE_TENSOR is None:
        print("CRITICAL: Failed to load and preprocess the image. The Flask app might not display predictions correctly.")
    if MODEL is None:
        print("CRITICAL: Failed to load the model. The Flask app might not display predictions correctly.")
    
    print("Starting Flask development server...")
    app.run(debug=True, use_reloader=False)
