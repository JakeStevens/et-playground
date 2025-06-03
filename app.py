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

# Define Main Inference Function
def get_inference_results():
    """
    Performs all inference steps: image loading, model loading, warm-up,
    benchmarking, and prediction.
    Returns a dictionary with all results or error information.
    """
    preprocessed_image_tensor = load_and_preprocess_image()
    model = load_model()

    if model is None or preprocessed_image_tensor is None:
        error_message = "Failed to load model" if model is None else "Failed to load and preprocess image"
        if model is None and preprocessed_image_tensor is None:
            error_message = "Failed to load model and image"
        print(f"Error in get_inference_results: {error_message}")
        return {
            'error': error_message,
            'top_5_predictions': [],
            'avg_latency': None,
            'min_latency': None,
            'max_latency': None,
            'std_latency': None,
            'image_url': IMAGE_URL  # Still provide IMAGE_URL for the template
        }

    # Perform warm-up runs
    print("Performing warm-up runs...")
    for _ in range(5):
        with torch.no_grad():
            model(preprocessed_image_tensor)
    print("Warm-up complete.")

    # Perform benchmarking
    print("Performing benchmarking...")
    avg_latency, min_latency, max_latency, std_latency = benchmark_inference(model, preprocessed_image_tensor)
    print(f"Benchmarking complete. Average latency: {avg_latency:.2f} ms")

    # Get top 5 predictions
    print("Generating top 5 predictions...")
    top_5_predictions = get_top5_predictions(model, preprocessed_image_tensor, IMAGENET_CLASSES)
    print("Top 5 predictions generated.")

    return {
        'top_5_predictions': top_5_predictions,
        'avg_latency': avg_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
        'std_latency': std_latency,
        'image_url': IMAGE_URL,
        'error': None
        # 'model': model, # also return model and tensor for global assignment - REMOVED
        # 'preprocessed_image_tensor': preprocessed_image_tensor - REMOVED
    }

# Removed global population of inference_results
# MODEL = None # Removed
# PREPROCESSED_IMAGE_TENSOR = None # Removed
# TOP_5_PREDICTIONS = [] # Removed
# AVG_LATENCY, MIN_LATENCY, MAX_LATENCY, STD_LATENCY = None, None, None, None # Removed


@app.route('/')
def index():
    results = get_inference_results()
    predictions_html = ""
    current_image_url = results.get('image_url', IMAGE_URL) # Use IMAGE_URL from constants as fallback

    if results['error']:
        predictions_html = f"<li>Error: {results['error']}</li>"
        avg_latency_html = "N/A"
        min_latency_html = "N/A"
        max_latency_html = "N/A"
        std_latency_html = "N/A"
    else:
        if results['top_5_predictions']:
            for pred in results['top_5_predictions']:
                if isinstance(pred, dict) and 'name' in pred and 'prob' in pred:
                    predictions_html += f"<li>Class Name: {pred['name']}, Probability: {pred['prob']*100:.4f}%</li>"
                else:
                    predictions_html += "<li>Invalid prediction format.</li>"
        else:
            predictions_html = "<li>No predictions available.</li>"

        avg_latency_html = f"{results['avg_latency']:.2f} ms" if results['avg_latency'] is not None else "N/A"
        min_latency_html = f"{results['min_latency']:.2f} ms" if results['min_latency'] is not None else "N/A"
        max_latency_html = f"{results['max_latency']:.2f} ms" if results['max_latency'] is not None else "N/A"
        std_latency_html = f"{results['std_latency']:.2f} ms" if results['std_latency'] is not None else "N/A"

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
                <img src="{current_image_url}" alt="Inference Image">
                <p style="text-align:center;">Image URL used for inference: <a href="{current_image_url}" target="_blank">{current_image_url}</a></p>
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
    # Critical error checks removed as errors are handled per request in index()
    print("Starting Flask development server...")
    app.run(debug=True, use_reloader=False)
