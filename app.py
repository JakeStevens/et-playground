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
def benchmark_inference(model, image_tensor, progress_callback):
    """Performs inference 1000 times, reports progress, and returns latency statistics."""
    latencies = []
    total_iterations = 1000

    if not model or not image_tensor:
        raise ValueError("Model or image tensor not provided to benchmark_inference")

    for i in range(total_iterations):
        start_time = time.time()
        with torch.no_grad():
            model(image_tensor)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

        current_progress = int(((i + 1) / total_iterations) * 100)
        if progress_callback:
            progress_callback(current_progress)

    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    std_latency = np.std(latencies)

    return {
        'avg_latency': avg_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
        'std_latency': std_latency
    }

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

class BenchmarkManager:
    def __init__(self):
        self.status = 'idle'  # idle, loading, running, complete, error
        self.progress = 0
        self.results = None
        self.error_message = None
        self.model = None
        self.image_tensor = None
        self.lock = threading.Lock()

    def _log(self, message):
        # Simple logger, replace with Flask app logger if available/preferred
        print(f"[BenchmarkManager] {message}")

    def load_resources_if_needed(self):
        with self.lock: # Ensure thread-safe check and load
            if self.model and self.image_tensor:
                self._log("Model and image tensor already loaded.")
                return True

            # If not loaded, set status to loading
            self.status = 'loading'
            self.progress = 0 # Reset progress for loading phase
            self.error_message = None

        self._log("Attempting to load model and image tensor...")
        try:
            # Temporarily release lock for potentially long I/O operations
            # This is a design choice: if another request comes, it might also try to load.
            # A more complex setup might use a dedicated loading lock or ensure only one loading attempt.

            current_model = load_model() # from app.py global scope
            current_image_tensor = load_and_preprocess_image() # from app.py global scope

            with self.lock:
                if current_model and current_image_tensor:
                    self.model = current_model
                    self.image_tensor = current_image_tensor
                    # Perform warm-up runs after loading model and tensor
                    self._log("Performing warm-up runs...")
                    for _ in range(5): # Warm-up runs
                        with torch.no_grad():
                            self.model(self.image_tensor)
                    self._log("Warm-up complete.")
                    # If status was 'loading', reset to 'idle' as resources are ready for benchmark/prediction
                    if self.status == 'loading':
                        self.status = 'idle'
                    self._log("Model and image tensor loaded successfully.")
                    return True
                else:
                    self.model = None
                    self.image_tensor = None
                    error_msg = "Failed to load model or image tensor."
                    if not current_model: error_msg += " Model loading failed."
                    if not current_image_tensor: error_msg += " Image tensor loading failed."
                    self.error_message = error_msg
                    self.status = 'error'
                    self._log(error_msg)
                    return False
        except Exception as e:
            with self.lock:
                self.model = None
                self.image_tensor = None
                self.error_message = f"Exception during resource loading: {str(e)}"
                self.status = 'error'
                self._log(self.error_message)
            return False

    def get_initial_predictions(self):
        self._log("Getting initial predictions...")
        if not self.load_resources_if_needed():
            with self.lock: # Read consistent error state
                return {'error': self.error_message, 'top_5_predictions': [], 'image_url': IMAGE_URL}

        if not self.model or not self.image_tensor: # Should be caught by load_resources_if_needed
            return {'error': "Model/Tensor not available after load attempt.", 'top_5_predictions': [], 'image_url': IMAGE_URL}

        self._log("Generating top 5 predictions...")
        # get_top5_predictions is still a global function in app.py
        predictions = get_top5_predictions(self.model, self.image_tensor, IMAGENET_CLASSES)
        self._log("Top 5 predictions generated.")
        return {'error': None, 'top_5_predictions': predictions, 'image_url': IMAGE_URL}

    def _update_progress_callback(self, current_percentage):
        with self.lock:
            if self.status == 'running': # Only update if still in running state
                self.progress = current_percentage

    def _perform_benchmark_and_update_status(self):
        self._log("Benchmark thread started.")
        try:
            # benchmark_inference is now the refactored global function in app.py
            latency_results = benchmark_inference(self.model, self.image_tensor, self._update_progress_callback)
            with self.lock:
                self.results = latency_results
                self.status = 'complete'
                self.progress = 100 # Ensure progress is 100% on completion
                self._log("Benchmark completed successfully.")
        except Exception as e:
            self._log(f"Exception in benchmark thread: {str(e)}")
            with self.lock:
                self.status = 'error'
                self.error_message = str(e)
                self.results = None # Clear any partial results

    def start_benchmark(self):
        self._log("Attempting to start benchmark...")
        with self.lock:
            if self.status == 'running' or self.status == 'loading':
                self._log(f"Cannot start benchmark, current status: {self.status}")
                return {'message': f'Benchmark process is already active ({self.status}). Please wait.'}, False

            # Attempt to load resources first if not already loaded
            # This call will acquire and release the lock internally

        if not self.load_resources_if_needed(): # This method handles its own logging and status updates on failure
            self._log("Resource loading failed prior to starting benchmark.")
            # get_status() will reflect the error from load_resources_if_needed
            return {'message': 'Failed to load resources for benchmarking.', 'error': self.error_message}, False

        # Now proceed with starting the benchmark thread
        with self.lock:
            self.status = 'running'
            self.progress = 0
            self.results = None
            self.error_message = None
            self._log("Benchmark status set to 'running'. Starting thread.")

        thread = threading.Thread(target=self._perform_benchmark_and_update_status)
        thread.daemon = True
        thread.start()

        return {'message': 'Benchmark started.'}, True

    def get_status(self):
        with self.lock:
            # Return a copy to prevent modification of internal state if the dict is manipulated by caller
            return {
                'status': self.status,
                'progress': self.progress,
                'results': self.results,
                'error_message': self.error_message
            }

# Global instance of the manager
benchmark_manager = BenchmarkManager()

@app.route('/start_benchmark', methods=['POST'])
def start_benchmark_route():
    message, success = benchmark_manager.start_benchmark()
    if success:
        return jsonify(message), 200
    else:
        # Consider appropriate status code for failure to start
        # 409 (Conflict) if already running/loading, 500 or 503 if resource loading failed
        status_code = 409 if benchmark_manager.get_status()['status'] in ['running', 'loading'] else 503
        return jsonify(message), status_code

@app.route('/benchmark_status')
def benchmark_status_route():
    try:
        # Ensure app.logger is available and configured.
        # For basic Flask apps, app.logger is available after app creation.
        # If running this snippet standalone, app.logger might need explicit setup,
        # but in the context of the existing app.py, it should be fine.

        status_data = benchmark_manager.get_status()

        # Log the data before jsonify
        # Use a print statement for simplicity if app.logger is problematic in the subtask environment,
        # but app.logger is preferred for Flask apps.
        print(f"INFO: Raw status data from BenchmarkManager: {status_data}")
        # Fallback to print if app.logger is not configured in this execution context:
        # try:
        #     app.logger.info(f"Raw status data from BenchmarkManager: {status_data}")
        # except AttributeError: # pragma: no cover
        #     print(f"INFO: Raw status data from BenchmarkManager: {status_data}")


        if not isinstance(status_data, dict):
            error_msg = f"BenchmarkManager.get_status() returned non-dict type: {type(status_data)}, value: {status_data}"
            print(f"ERROR: {error_msg}")
            # try:
            #    app.logger.error(error_msg)
            # except AttributeError: # pragma: no cover
            #    print(f"ERROR: {error_msg}")
            return jsonify({'error': 'Internal server error: Invalid status data format', 'status': 'error'}), 500

        if 'status' not in status_data:
            error_msg = f"Key 'status' is missing from BenchmarkManager status_data: {status_data}"
            print(f"ERROR: {error_msg}")
            # try:
            #    app.logger.error(error_msg)
            # except AttributeError: # pragma: no cover
            #    print(f"ERROR: {error_msg}")

            # Defensively add status and error message
            status_data['status'] = 'error'
            status_data['error_message'] = str(status_data.get('error_message', '')) + " [System Error: Status key was missing from server data]"
            # Also ensure other essential keys for JS are present if status is error
            status_data.setdefault('progress', 0)
            status_data.setdefault('results', None)


        return jsonify(status_data)
    except Exception as e:
        # Log stack trace
        import traceback
        error_details = f"Error in /benchmark_status route: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_details}")
        # try:
        #    app.logger.error(f"Error in /benchmark_status route: {str(e)}", exc_info=True)
        # except AttributeError: # pragma: no cover
        #    print(f"ERROR: Error in /benchmark_status route: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error while fetching status', 'details': str(e), 'status': 'error'}), 500

@app.route('/')
def index():
    # Use the manager to get initial predictions
    results = benchmark_manager.get_initial_predictions()

    current_image_url = results.get('image_url', IMAGE_URL) # IMAGE_URL is still a global constant
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
                        if (!data || typeof data.status === 'undefined') {
                            console.error('Invalid or incomplete status data received:', data);
                            statusMessage.textContent = 'Error: Received invalid status data from server.';
                            statusMessage.style.color = 'red';
                            if (benchmarkInterval) clearInterval(benchmarkInterval);
                            button.disabled = false;
                            button.textContent = 'Start Benchmark'; // Or 'Re-run Benchmark' if appropriate
                            progressBarContainer.style.display = 'none';
                            return;
                        }
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
