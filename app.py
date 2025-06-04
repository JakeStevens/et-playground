from flask import Flask, jsonify # Add jsonify
import threading
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

# --- Global variables for benchmarking state ---
benchmark_status = {
    'status': 'idle',  # idle, running, complete, error
    'progress': 0,    # 0-100
    'results': None,
    'error_message': None
}
benchmark_lock = threading.Lock()
# --- Store model and tensor for benchmark thread ---
# These will be populated by the main route before benchmark starts
_global_model_for_benchmark = None
_global_tensor_for_benchmark = None
# --- End Global variables ---

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
    """Performs inference 1000 times and returns latency statistics, updating global status."""
    global benchmark_status, benchmark_lock
    latencies = []
    total_iterations = 1000 # Defined for clarity

    try:
        for i in range(total_iterations):
            start_time = time.time()
            with torch.no_grad():
                model(image_tensor)
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # Update progress
            current_progress = int(((i + 1) / total_iterations) * 100)
            with benchmark_lock:
                benchmark_status['progress'] = current_progress

        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        std_latency = np.std(latencies)

        with benchmark_lock:
            benchmark_status['results'] = {
                'avg_latency': avg_latency,
                'min_latency': min_latency,
                'max_latency': max_latency,
                'std_latency': std_latency
            }
            # Status will be set to 'complete' by the calling thread function
        return avg_latency, min_latency, max_latency, std_latency # Still return for potential direct use or other callers
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        with benchmark_lock:
            benchmark_status['status'] = 'error'
            benchmark_status['error_message'] = str(e)
            benchmark_status['results'] = None
        # Re-raise or return error indication if needed by other parts of the app not using global status
        raise # Or return None, None, None, None

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
    Performs image loading, model loading, warm-up, and top-5 prediction.
    Stores model and tensor globally for benchmark thread.
    Benchmarking is now handled separately.
    Returns a dictionary with prediction results or error information.
    """
    global benchmark_status, benchmark_lock, _global_model_for_benchmark, _global_tensor_for_benchmark # Added globals

    preprocessed_image_tensor = load_and_preprocess_image()
    model = load_model()

    if model is None or preprocessed_image_tensor is None:
        error_message = "Failed to load model" if model is None else "Failed to load and preprocess image"
        if model is None and preprocessed_image_tensor is None:
            error_message = "Failed to load model and image"
        print(f"Error in get_inference_results: {error_message}")
        # Update global status if there's an error here that prevents even predictions
        with benchmark_lock:
            benchmark_status['status'] = 'error'
            benchmark_status['error_message'] = error_message
        # Also ensure global model/tensor are None if there's an error
        _global_model_for_benchmark = None
        _global_tensor_for_benchmark = None
        return {
            'error': error_message,
            'top_5_predictions': [],
            'image_url': IMAGE_URL
        }

    # Store model and tensor globally
    _global_model_for_benchmark = model
    _global_tensor_for_benchmark = preprocessed_image_tensor

    # Perform warm-up runs
    print("Performing warm-up runs...")
    for _ in range(5): # Reduced warm-up for quicker UI response, adjust as needed
        with torch.no_grad():
            model(preprocessed_image_tensor)
    print("Warm-up complete.")

    # Get top 5 predictions
    print("Generating top 5 predictions...")
    top_5_predictions = get_top5_predictions(model, preprocessed_image_tensor, IMAGENET_CLASSES)
    print("Top 5 predictions generated.")

    # Benchmarking is no longer called here directly.
    # The main results dictionary will not contain latency stats initially.
    # These will be fetched via the /benchmark_status endpoint.
    return {
        'top_5_predictions': top_5_predictions,
        'image_url': IMAGE_URL,
        'error': None
        # No longer returning _model_for_benchmark and _tensor_for_benchmark in dict
    }

# Removed global population of inference_results
# MODEL = None # Removed
# PREPROCESSED_IMAGE_TENSOR = None # Removed
# TOP_5_PREDICTIONS = [] # Removed
# AVG_LATENCY, MIN_LATENCY, MAX_LATENCY, STD_LATENCY = None, None, None, None # Removed

def _run_benchmark_thread():
    """Wrapper function to run benchmark_inference in a thread and update status."""
    global benchmark_status, benchmark_lock, _global_model_for_benchmark, _global_tensor_for_benchmark

    if not _global_model_for_benchmark or not _global_tensor_for_benchmark:
        print("Error: Model or tensor not available for benchmarking.")
        with benchmark_lock:
            benchmark_status['status'] = 'error'
            benchmark_status['error_message'] = 'Model or tensor not loaded for benchmark.'
        return

    try:
        print("Benchmark thread started...")
        # benchmark_inference will update progress and results internally using global benchmark_status
        benchmark_inference(_global_model_for_benchmark, _global_tensor_for_benchmark)
        with benchmark_lock:
            if benchmark_status['status'] != 'error': # Don't override error status if benchmark_inference set it
                benchmark_status['status'] = 'complete'
        print("Benchmark thread finished.")
    except Exception as e:
        print(f"Exception in benchmark thread: {e}")
        with benchmark_lock:
            benchmark_status['status'] = 'error'
            benchmark_status['error_message'] = str(e)

@app.route('/start_benchmark', methods=['POST'])
def start_benchmark_route():
    global benchmark_status, benchmark_lock, _global_model_for_benchmark, _global_tensor_for_benchmark

    with benchmark_lock:
        if benchmark_status['status'] == 'running':
            return jsonify({'message': 'Benchmark is already running.'}), 409 # Conflict

        # Reset status for a new run
        benchmark_status['status'] = 'running'
        benchmark_status['progress'] = 0
        benchmark_status['results'] = None
        benchmark_status['error_message'] = None

    # The _global_model_for_benchmark and _global_tensor_for_benchmark
    # are expected to be populated by the initial call to get_inference_results()
    # from the main '/' route. This is a design choice from the previous step.
    # If they are not populated, the _run_benchmark_thread will handle it.

    thread = threading.Thread(target=_run_benchmark_thread)
    thread.daemon = True # Allows main program to exit even if threads are still running
    thread.start()

    return jsonify({'message': 'Benchmark started.'})

@app.route('/benchmark_status')
def benchmark_status_route():
    global benchmark_status, benchmark_lock
    with benchmark_lock:
        # Make a copy to avoid issues if the status is updated while sending response
        status_copy = dict(benchmark_status)
    return jsonify(status_copy)

@app.route('/')
def index():
    results = get_inference_results() # This loads model/tensor globally and gets initial predictions

    current_image_url = results.get('image_url', IMAGE_URL)
    predictions_html = ""

    if results.get('error') and not results.get('top_5_predictions'):
        predictions_html = f"<li>Error fetching initial predictions: {results['error']}</li>"
    elif not results.get('top_5_predictions'):
        predictions_html = "<li>No initial predictions available.</li>"
    else:
        for pred in results['top_5_predictions']:
            if isinstance(pred, dict) and 'name' in pred and 'prob' in pred:
                predictions_html += f"<li>Class Name: {pred['name']}, Probability: {pred['prob']*100:.4f}%</li>"
            else:
                predictions_html += "<li>Invalid prediction format.</li>"

    # Initial latency values - these will be updated by JS after benchmark
    avg_latency_html = "N/A (Run benchmark)"
    min_latency_html = "N/A (Run benchmark)"
    max_latency_html = "N/A (Run benchmark)"
    std_latency_html = "N/A (Run benchmark)"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
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
            .progress-bar-container {{ width: 100%; background-color: #e0e0e0; border-radius: 4px; margin-bottom: 10px; overflow: hidden; }}
            .progress-bar {{ width: 0%; height: 20px; background-color: #4caf50; text-align: center; line-height: 20px; color: white; border-radius: 4px; transition: width 0.3s ease, background-color 0.3s ease; }}
            #benchmarkStatusMessage {{ margin-bottom: 10px; font-style: italic; }}
            button#startBenchmarkButton {{ padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; transition: background-color 0.3s ease; }}
            button#startBenchmarkButton:disabled {{ background-color: #ccc; cursor: not-allowed; }}
            button#startBenchmarkButton:hover:not(:disabled) {{ background-color: #0056b3; }}
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
                <button id="startBenchmarkButton" onclick="startBenchmark()">Start Benchmark</button>
                <div id="benchmarkStatusMessage" style="display:none; margin-top:10px;"></div>
                <div class="progress-bar-container" id="progressBarContainer" style="display:none; margin-top:5px;">
                    <div class="progress-bar" id="progressBar">0%</div>
                </div>
                <p>Average Latency: <span id="avgLatency">{avg_latency_html}</span></p>
                <p>Minimum Latency: <span id="minLatency">{min_latency_html}</span></p>
                <p>Maximum Latency: <span id="maxLatency">{max_latency_html}</span></p>
                <p>Standard Deviation: <span id="stdLatency">{std_latency_html}</span></p>
            </div>
        </div>

        <script>
            let benchmarkInterval;

            function startBenchmark() {{
                const button = document.getElementById('startBenchmarkButton');
                const statusMessage = document.getElementById('benchmarkStatusMessage');
                const progressBarContainer = document.getElementById('progressBarContainer');
                const progressBar = document.getElementById('progressBar');

                button.disabled = true;
                button.textContent = 'Benchmark Running...';
                statusMessage.textContent = 'Initializing benchmark...';
                statusMessage.style.display = 'block';
                statusMessage.style.color = 'inherit'; // Reset color
                progressBarContainer.style.display = 'block';
                progressBar.style.width = '0%';
                progressBar.textContent = '0%';
                progressBar.style.backgroundColor = '#4caf50'; // Reset color

                document.getElementById('avgLatency').textContent = 'N/A';
                document.getElementById('minLatency').textContent = 'N/A';
                document.getElementById('maxLatency').textContent = 'N/A';
                document.getElementById('stdLatency').textContent = 'N/A';

                fetch('/start_benchmark', {{ method: 'POST' }})
                    .then(response => {{
                        if (!response.ok) {{
                            // Try to parse error message from server if available
                            return response.json().then(err => {{
                                throw new Error(err.message || `Server error: ${response.status}`);
                            }}).catch(() => {{
                                // Fallback if parsing error message fails
                                throw new Error(`Server error: ${response.status}`);
                            }});
                        }}
                        return response.json();
                    }})
                    .then(data => {{
                        statusMessage.textContent = 'Benchmark running...';
                        console.log(data.message);
                        if (benchmarkInterval) clearInterval(benchmarkInterval);
                        benchmarkInterval = setInterval(getBenchmarkStatus, 1000);
                    }})
                    .catch(error => {{
                        console.error('Error starting benchmark:', error);
                        statusMessage.textContent = 'Error: ' + error.message;
                        statusMessage.style.color = 'red';
                        button.disabled = false;
                        button.textContent = 'Start Benchmark';
                        progressBarContainer.style.display = 'none';
                    }});
            }}

            function getBenchmarkStatus() {{
                const statusMessage = document.getElementById('benchmarkStatusMessage');
                const progressBar = document.getElementById('progressBar');
                const progressBarContainer = document.getElementById('progressBarContainer');
                const button = document.getElementById('startBenchmarkButton');

                fetch('/benchmark_status')
                    .then(response => {{
                         if (!response.ok) {{
                            return response.json().then(err => {{
                                throw new Error(err.message || `Server error: ${response.status}`);
                            }}).catch(() => {{
                                throw new Error(`Server error: ${response.status}`);
                            }});
                        }}
                        return response.json();
                    }})
                    .then(data => {{
                        console.log('Status update:', data); // For debugging
                        if (data.status === 'running') {{
                            progressBar.style.width = data.progress + '%';
                            progressBar.textContent = data.progress + '%';
                            statusMessage.textContent = 'Benchmark running... (' + data.progress + '%)';
                            progressBar.style.backgroundColor = '#4caf50'; // Green for running
                        }} else if (data.status === 'complete') {{
                            clearInterval(benchmarkInterval);
                            statusMessage.textContent = 'Benchmark complete!';
                            statusMessage.style.color = 'green';
                            progressBar.style.width = '100%';
                            progressBar.textContent = '100%';
                            progressBar.style.backgroundColor = '#4caf50'; // Green for complete
                            button.disabled = false;
                            button.textContent = 'Re-run Benchmark';
                            if (data.results) {{
                                document.getElementById('avgLatency').textContent = data.results.avg_latency !== null ? data.results.avg_latency.toFixed(2) + ' ms' : 'N/A';
                                document.getElementById('minLatency').textContent = data.results.min_latency !== null ? data.results.min_latency.toFixed(2) + ' ms' : 'N/A';
                                document.getElementById('maxLatency').textContent = data.results.max_latency !== null ? data.results.max_latency.toFixed(2) + ' ms' : 'N/A';
                                document.getElementById('stdLatency').textContent = data.results.std_latency !== null ? data.results.std_latency.toFixed(2) + ' ms' : 'N/A';
                            }}
                        }} else if (data.status === 'error') {{
                            clearInterval(benchmarkInterval);
                            statusMessage.textContent = 'Benchmark error: ' + (data.error_message || 'Unknown error');
                            statusMessage.style.color = 'red';
                            progressBar.style.backgroundColor = 'red'; // Red for error
                            progressBar.style.width = '100%'; // Show full bar but in red
                            progressBar.textContent = 'Error';
                            //progressBarContainer.style.display = 'none'; // Optionally hide or show error state
                            button.disabled = false;
                            button.textContent = 'Re-run Benchmark';
                        }} else if (data.status === 'idle') {{
                            clearInterval(benchmarkInterval);
                            // Only hide if it was not an explicit error or completion
                            if(statusMessage.style.color !== 'red' && statusMessage.style.color !== 'green'){{
                               statusMessage.style.display = 'none';
                            }}
                            progressBarContainer.style.display = 'none';
                            button.disabled = false;
                            button.textContent = 'Start Benchmark';
                        }}
                    }})
                    .catch(error => {{
                        console.error('Error fetching benchmark status:', error);
                        clearInterval(benchmarkInterval);
                        statusMessage.textContent = 'Error fetching status: ' + error.message;
                        statusMessage.style.color = 'red';
                        button.disabled = false;
                        button.textContent = 'Start Benchmark';
                        progressBarContainer.style.display = 'none';
                    }});
            }}

            // Optional: If you want to check status on page load (e.g. if benchmark was started in another tab)
            // window.addEventListener('DOMContentLoaded', (event) => {{
            //     getBenchmarkStatus();
            // }});
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == '__main__':
    # Critical error checks removed as errors are handled per request in index()
    print("Starting Flask development server...")
    app.run(debug=True, use_reloader=False)
