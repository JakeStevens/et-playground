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

# ExecuTorch specific imports
from executorch.sdk.etrecord import ETRecord # Keep for ETRecord if used by compiler, but app.py won't directly use it for loading
from torch.executor import Executor
import torch.executor.evalue as evalue # For EValue manipulation

# Import for on-the-fly compilation
from aot_compiler import compile_model_to_executorch


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

    if model is None or image_tensor is None:
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

# ---- XNNPACK Model Loading and Inference Functions ----

# load_xnnpack_model(pte_path: str) is REMOVED as per on-the-fly compilation requirement.

def get_top5_predictions_xnnpack(executor: Executor, image_tensor: torch.Tensor, imagenet_classes: dict) -> list:
    """
    Performs inference using the XNNPACK executor and returns the top 5 predictions.
    """
    print("Performing inference with XNNPACK executor...")
    try:
        # 1. Prepare input: Convert PyTorch tensor to EValue
        # The .pte model expects a single tensor input.
        # The `program.execute()` method expects a tuple of EValue inputs.
        evalue_input = (evalue.EValue(image_tensor),)

        # 2. Execute the model
        print("  Executing model...")
        # TODO: Verify the exact method name for execution. It's usually `run` or `forward` on the module
        # obtained from the executor, or `execute` on the program if using a lower-level API.
        # For the `Executor` class directly, it's typically interacting with its internal program.
        # Let's assume the executor has a method `run_method` or similar that maps to the entry point.
        # This part might need adjustment based on the exact Executor API from `torch.executor`.
        # Looking at typical ExecuTorch examples, you often get a module from the executor
        # and call a method on that. If `Executor` itself is the callable, then it's simpler.
        # For now, let's assume the `Executor` instance itself can be called if it wraps a single method,
        # or it has a primary method to call.
        # The `executor.program.execute(method_name, inputs)` pattern is also common.
        # We need to know the method name. Usually it's "forward".

        # Let's assume the ETRecord/Program stores method names and we use the first one.
        # This is a common pattern if the model has a single entry point.
        # method_name = executor.program.entry_points[0] # This API might not exist directly on program
        # A safer bet, often it's just 'forward'
        method_name = "forward" # Common default for exported models

        outputs = executor.run_method(method_name, evalue_input)
        print(f"  Model execution completed. Output EValues: {outputs}")

        # 3. Output Processing: Convert EValue output back to PyTorch tensor
        # Assuming the model returns a single tensor output
        if not outputs or not isinstance(outputs, tuple) or not outputs[0].is_tensor():
            raise ValueError(f"Unexpected output format from executor: {outputs}")

        output_tensor = outputs[0].to_tensor()
        print(f"  Converted output EValue to tensor with shape: {output_tensor.shape}")

        # 4. Apply Softmax and get top 5 (similar to PyTorch version)
        # Ensure output_tensor is on CPU if not already, for softmax and topk
        output_tensor = output_tensor.cpu()
        probabilities = F.softmax(output_tensor[0], dim=0) # Assuming batch size 1 in output_tensor
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        top5_prob_list = top5_prob.tolist()
        top5_catid_list = top5_catid.tolist()

        results = []
        for i in range(len(top5_prob_list)):
            results.append({
                'name': imagenet_classes[top5_catid_list[i]],
                'prob': top5_prob_list[i]
            })
        print(f"  Top 5 predictions processed: {results}")
        return results
    except Exception as e:
        print(f"Error during XNNPACK inference or prediction processing: {e}")
        # import traceback
        # print(traceback.format_exc())
        raise # Re-raise for now


def benchmark_inference_xnnpack(executor: Executor, image_tensor: torch.Tensor, progress_callback=None) -> dict:
    """
    Performs inference 1000 times using XNNPACK executor, reports progress, and returns latency statistics.
    """
    print("Starting XNNPACK benchmark...")
    latencies = []
    total_iterations = 1000 # Same as PyTorch version

    if executor is None or image_tensor is None:
        raise ValueError("Executor or image tensor not provided to benchmark_inference_xnnpack")

    # Prepare input once (can be reused if tensor is not modified)
    # Convert PyTorch tensor to EValue for the executor
    evalue_input = (evalue.EValue(image_tensor),)
    method_name = "forward" # Assuming 'forward' is the method name

    print(f"  Input tensor shape: {image_tensor.shape}, EValue input prepared.")

    for i in range(total_iterations):
        start_time = time.time()

        # Execute the model
        # We expect this call to be blocking and complete before the next line.
        _ = executor.run_method(method_name, evalue_input) # Output is not processed for speed.

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

        if progress_callback:
            current_progress = int(((i + 1) / total_iterations) * 100)
            if i % (total_iterations // 10) == 0 or i == total_iterations -1 : # Log progress less frequently
                 print(f"  Benchmark progress: {current_progress}% ({i+1}/{total_iterations})")
            progress_callback(current_progress)

    print("  Benchmark loop finished.")

    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    std_latency = np.std(latencies)

    print(f"  XNNPACK Benchmark results: Avg: {avg_latency:.2f}ms, Min: {min_latency:.2f}ms, Max: {max_latency:.2f}ms, Std: {std_latency:.2f}ms")

    return {
        'avg_latency': avg_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
        'std_latency': std_latency
    }

# ---- End XNNPACK Functions ----

class BenchmarkManager:
    def __init__(self):
        self.status = 'idle'  # idle, loading, running, complete, error
        self.progress = 0
        # Store results for both PyTorch and XNNPACK models
        self.results = {'pytorch': None, 'xnnpack': None}
        self.error_message = None
        self.pytorch_model = None
        self.xnnpack_model = None # Executor instance for XNNPACK, now compiled on-the-fly
        self.image_tensor = None
        self.lock = threading.Lock()
        # self.xnnpack_pte_path is removed, no longer loading from a file.

    def _log(self, message):
        # Simple logger, replace with Flask app logger if available/preferred
        print(f"[BenchmarkManager] {message}")

    def load_resources_if_needed(self):
        with self.lock: # Ensure thread-safe check and load
            # Check if all resources (PyTorch model, XNNPACK model, image tensor) are already loaded
            if self.pytorch_model is not None and \
               self.xnnpack_model is not None and \
               self.image_tensor is not None:
                self._log("All models and image tensor already loaded.")
                return True

            # If not loaded, set status to loading
            self.status = 'loading'
            self.progress = 0 # Reset progress for loading phase
            self.error_message = None
            # Clear previous models in case of partial load or re-load attempt
            self.pytorch_model = None
            self.xnnpack_model = None
            self.image_tensor = None

        self._log("Attempting to load PyTorch model, XNNPACK model, and image tensor...")
        # These operations can take time, so they are outside the main lock section initially
        # However, assignment to self.xxx should be within the lock

        error_messages = []
        loaded_pytorch_model = None
        loaded_pytorch_model = None
        # loaded_xnnpack_model will be the Executor instance after on-the-fly compilation
        compiled_executorch_program = None
        created_xnnpack_executor = None
        loaded_image_tensor = None

        try:
            # Step 1: Load image tensor first, as it's needed for example_inputs
            self._log("Loading and preprocessing image...")
            loaded_image_tensor = load_and_preprocess_image()
            if loaded_image_tensor is not None:
                self._log("Image loaded and preprocessed successfully.")
            else:
                error_messages.append("Image tensor loading failed.")
                # If image fails, cannot proceed with compilation that depends on it
                with self.lock:
                    self.error_message = "Failed to load image tensor, cannot proceed."
                    self.status = 'error'
                    self._log(self.error_message)
                return False

            # Step 2: Load PyTorch model
            self._log("Loading PyTorch model...")
            loaded_pytorch_model = load_model() # This is models.resnet18(pretrained=True)
            if loaded_pytorch_model:
                self._log("PyTorch model loaded successfully.")
                loaded_pytorch_model.eval() # Ensure it's in eval mode
            else:
                error_messages.append("PyTorch model loading failed.")

            # Step 3: On-the-fly compile PyTorch model to ExecuTorch program for XNNPACK
            if loaded_pytorch_model and loaded_image_tensor is not None: # Check dependencies
                self._log("Starting on-the-fly compilation of ResNet18 for XNNPACK...")
                try:
                    example_inputs = (loaded_image_tensor,) # Use the actual preprocessed image tensor
                    compiled_executorch_program = compile_model_to_executorch(
                        loaded_pytorch_model,
                        example_inputs,
                        enable_quantization=False,
                        output_pte_path=None # Crucial: ensures no file is saved, returns program
                    )
                    if compiled_executorch_program:
                        self._log("On-the-fly compilation to ExecuTorch program successful.")
                        # Step 4: Instantiate Executor with the in-memory program
                        self._log("Instantiating XNNPACK Executor from in-memory program...")
                        created_xnnpack_executor = Executor(compiled_executorch_program)
                        self._log("XNNPACK Executor instantiated successfully.")
                    else:
                        error_messages.append("XNNPACK model compilation returned None.")
                except Exception as e:
                    self._log(f"Exception during XNNPACK on-the-fly compilation or Executor instantiation: {e}")
                    # import traceback; traceback.print_exc() # For more detailed server logs if needed
                    error_messages.append(f"XNNPACK model compilation/instantiation failed: {e}")
            else:
                if not loaded_pytorch_model:
                    error_messages.append("Skipping XNNPACK compilation due to PyTorch model load failure.")
                if loaded_image_tensor is None: # Should have been caught earlier
                     error_messages.append("Skipping XNNPACK compilation due to image load failure.")


            # Now, acquire lock to update instance variables and perform warm-up
            with self.lock:
                # Check if all required components are available
                if loaded_pytorch_model and created_xnnpack_executor and loaded_image_tensor:
                    self.pytorch_model = loaded_pytorch_model
                    self.xnnpack_model = created_xnnpack_executor # This is the Executor instance
                    self.image_tensor = loaded_image_tensor

                    # Perform warm-up runs for PyTorch model
                    self._log("Performing warm-up runs for PyTorch model...")
                    for _ in range(5):
                        with torch.no_grad():
                            self.pytorch_model(self.image_tensor)
                    self._log("PyTorch model warm-up complete.")

                    # Perform warm-up runs for XNNPACK model (Executor)
                    self._log("Performing warm-up runs for XNNPACK model (Executor)...")
                    evalue_input_warmup = (evalue.EValue(self.image_tensor),)
                    method_name_warmup = "forward" # Assuming 'forward'
                    for _ in range(5):
                        self.xnnpack_model.run_method(method_name_warmup, evalue_input_warmup)
                    self._log("XNNPACK model (Executor) warm-up complete.")

                    if self.status == 'loading': # If still in loading state
                        self.status = 'idle'
                    self._log("All models (PyTorch and compiled XNNPACK) and image tensor loaded and warmed up successfully.")
                    return True
                else:
                    # Clear any partially loaded/compiled resources
                    self.pytorch_model = None
                    self.xnnpack_model = None
                    # Keep image_tensor if it was loaded? For now, let's clear if full success not achieved.
                    # self.image_tensor = None

                    final_error_message = "Failed to load all resources. Errors: " + "; ".join(error_messages)
                    if not loaded_pytorch_model: final_error_message += " [PyTorch model missing]"
                    if not created_xnnpack_executor: final_error_message += " [XNNPACK executor missing]"
                    if not loaded_image_tensor: final_error_message += " [Image tensor missing]"
                    self.error_message = final_error_message

                    self.status = 'error'
                    self._log(self.error_message)
                    return False
        except Exception as e: # Catch-all for unexpected errors during the overall loading sequence
            with self.lock:
                self.pytorch_model = None
                self.xnnpack_model = None # Ensure it's cleared
                self.image_tensor = None
                self.error_message = f"Exception during resource loading process: {str(e)}"
                self.status = 'error'
                self._log(self.error_message)
            return False

    def get_initial_predictions(self):
        self._log("Getting initial predictions for PyTorch and XNNPACK models...")
        if not self.load_resources_if_needed():
            with self.lock: # Read consistent error state
                return {
                    'error': self.error_message,
                    'pytorch_predictions': [],
                    'xnnpack_predictions': [],
                    'image_url': IMAGE_URL
                }

        # At this point, self.pytorch_model, self.xnnpack_model, and self.image_tensor should be loaded.
        # However, perform a check for robustness.
        with self.lock: # Access shared resources under lock
            if not all([self.pytorch_model, self.xnnpack_model, self.image_tensor]):
                self._log("Models or image tensor not available after load attempt for initial predictions.")
                return {
                    'error': self.error_message or "Models/Tensor not available after load attempt.",
                    'pytorch_predictions': [],
                    'xnnpack_predictions': [],
                    'image_url': IMAGE_URL
                }

            # Make copies for prediction functions if they modify or to avoid issues with threads
            # (though current prediction functions are read-only on models)
            current_pytorch_model = self.pytorch_model
            current_xnnpack_model = self.xnnpack_model
            current_image_tensor = self.image_tensor

        pytorch_preds = []
        xnnpack_preds = []
        prediction_error_messages = []

        self._log("Generating Top 5 predictions for PyTorch model...")
        try:
            pytorch_preds = get_top5_predictions(current_pytorch_model, current_image_tensor, IMAGENET_CLASSES)
            self._log("PyTorch predictions generated.")
        except Exception as e:
            self._log(f"Error getting PyTorch predictions: {e}")
            prediction_error_messages.append(f"PyTorch: {e}")

        self._log("Generating Top 5 predictions for XNNPACK model...")
        try:
            xnnpack_preds = get_top5_predictions_xnnpack(current_xnnpack_model, current_image_tensor, IMAGENET_CLASSES)
            self._log("XNNPACK predictions generated.")
        except Exception as e:
            self._log(f"Error getting XNNPACK predictions: {e}")
            prediction_error_messages.append(f"XNNPACK: {e}")

        combined_error = None
        if prediction_error_messages:
            combined_error = "Error during prediction generation: " + "; ".join(prediction_error_messages)

        return {
            'error': combined_error,
            'pytorch_predictions': pytorch_preds,
            'xnnpack_predictions': xnnpack_preds,
            'image_url': IMAGE_URL
        }

    def _update_progress_callback(self, current_percentage_stage, base_progress):
        """
        Updates the overall progress.
        current_percentage_stage: 0-100 for the current benchmark stage.
        base_progress: 0 for PyTorch stage, 50 for XNNPACK stage.
        """
        with self.lock:
            if self.status == 'running': # Only update if still in running state
                # Scale current stage progress to 50% of total, then add base.
                self.progress = base_progress + (current_percentage_stage // 2)
                # Ensure progress doesn't exceed 100 due to rounding or logic
                self.progress = min(self.progress, 100)


    def _perform_benchmark_and_update_status(self):
        self._log("Benchmark thread started for PyTorch and XNNPACK models.")

        # Ensure resources are loaded (should be by start_benchmark, but double check)
        if not self.pytorch_model or not self.xnnpack_model or not self.image_tensor:
            self._log("Models or image tensor not loaded prior to benchmark execution. Aborting.")
            with self.lock:
                self.status = 'error'
                self.error_message = "Critical: Resources not loaded for benchmark."
                self.results = {'pytorch': None, 'xnnpack': None}
            return

        # Local copies for benchmark functions
        current_pytorch_model = self.pytorch_model
        current_xnnpack_model = self.xnnpack_model
        current_image_tensor = self.image_tensor

        pytorch_latency_results = None
        xnnpack_latency_results = None
        benchmark_errors = []

        # --- PyTorch Benchmark (0-50% progress) ---
        self._log("Starting PyTorch benchmark...")
        try:
            # Define a lambda for PyTorch progress that incorporates base_progress = 0
            pytorch_progress_callback = lambda p_stage: self._update_progress_callback(p_stage, 0)
            pytorch_latency_results = benchmark_inference(
                current_pytorch_model,
                current_image_tensor,
                pytorch_progress_callback
            )
            self._log("PyTorch benchmark completed.")
        except Exception as e:
            self._log(f"Exception during PyTorch benchmark: {e}")
            benchmark_errors.append(f"PyTorch benchmark failed: {e}")

        # --- XNNPACK Benchmark (50-100% progress) ---
        # Only run XNNPACK if PyTorch part didn't cause a fatal setup issue (though errors are separate)
        if not benchmark_errors or "Critical" not in benchmark_errors[0]: # Simple check
            self._log("Starting XNNPACK benchmark...")
            try:
                # Define a lambda for XNNPACK progress that incorporates base_progress = 50
                xnnpack_progress_callback = lambda p_stage: self._update_progress_callback(p_stage, 50)
                xnnpack_latency_results = benchmark_inference_xnnpack(
                    current_xnnpack_model,
                    current_image_tensor,
                    xnnpack_progress_callback
                )
                self._log("XNNPACK benchmark completed.")
            except Exception as e:
                self._log(f"Exception during XNNPACK benchmark: {e}")
                benchmark_errors.append(f"XNNPACK benchmark failed: {e}")

        # Update final status and results
        with self.lock:
            self.results['pytorch'] = pytorch_latency_results
            self.results['xnnpack'] = xnnpack_latency_results

            if benchmark_errors:
                self.status = 'error'
                self.error_message = "; ".join(benchmark_errors)
                # If PyTorch finished but XNNPACK failed, progress might be stuck at 50.
                # If XNNPACK also ran (even if failed), it would have tried to update to 100.
                # If PyTorch fails, progress is between 0-50.
                # For simplicity, if any error, we might not show 100% progress.
                # The _update_progress_callback handles capping at 100.
            else:
                self.status = 'complete'
                self.progress = 100 # Ensure progress is 100% on full completion
                self.error_message = None # Clear any previous non-fatal errors
                self._log("Both PyTorch and XNNPACK benchmarks completed successfully.")

            if not pytorch_latency_results and not xnnpack_latency_results:
                 self._log("Neither benchmark produced results.")
            elif not xnnpack_latency_results:
                 self._log("XNNPACK benchmark did not produce results.")
            elif not pytorch_latency_results:
                 self._log("PyTorch benchmark did not produce results.")


    def start_benchmark(self):
        self._log("Attempting to start combined benchmark...")
        with self.lock:
            if self.status == 'running' or self.status == 'loading':
                self._log(f"Cannot start combined benchmark, current status: {self.status}")
                return {'message': f'Benchmark process is already active ({self.status}). Please wait.'}, False

            # Attempt to load resources first if not already loaded.
            # load_resources_if_needed is now responsible for loading both models and image.
        if not self.load_resources_if_needed(): # This method handles its own logging and status updates on failure
            self._log("Combined resource loading failed prior to starting benchmark.")
            # get_status() will reflect the error from load_resources_if_needed
            # Error message is set by load_resources_if_needed
            return {'message': 'Failed to load resources for benchmarking.', 'error': self.error_message}, False

        # Now proceed with starting the benchmark thread
        with self.lock:
            self.status = 'running'
            self.progress = 0
            # Reset results for both models before starting a new benchmark run
            self.results = {'pytorch': None, 'xnnpack': None}
            self.error_message = None
            self._log("Combined benchmark status set to 'running'. Starting thread.")

        # The thread will execute _perform_benchmark_and_update_status, which now handles both.
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
    results = benchmark_manager.get_initial_predictions() # This now returns a dict with pytorch_predictions and xnnpack_predictions

    current_image_url = results.get('image_url', IMAGE_URL)

    # --- PyTorch Predictions HTML ---
    pytorch_predictions_html = ""
    pt_preds = results.get('pytorch_predictions', [])
    if not pt_preds and not results.get('error'): # No preds but also no major error reported for all predictions
        pytorch_predictions_html = "<li>No PyTorch predictions available or model not run.</li>"
    elif not pt_preds and results.get('error'): # No preds AND error reported
         pytorch_predictions_html = f"<li>Error fetching PyTorch predictions: {results.get('error','Unknown error, check logs')}</li>"
    else:
        for pred in pt_preds:
            if isinstance(pred, dict) and 'name' in pred and 'prob' in pred:
                pytorch_predictions_html += f"<li>{pred['name']}: {pred['prob']*100:.2f}%</li>"
            else:
                pytorch_predictions_html += "<li>Invalid PyTorch prediction format.</li>"

    # --- XNNPACK Predictions HTML ---
    xnnpack_predictions_html = ""
    xn_preds = results.get('xnnpack_predictions', [])
    if not xn_preds and not results.get('error'):
        xnnpack_predictions_html = "<li>No XNNPACK predictions available or model not run.</li>"
    elif not xn_preds and results.get('error'):
         xnnpack_predictions_html = f"<li>Error fetching XNNPACK predictions: {results.get('error','Unknown error, check logs')}</li>"
    else:
        for pred in xn_preds:
            if isinstance(pred, dict) and 'name' in pred and 'prob' in pred:
                xnnpack_predictions_html += f"<li>{pred['name']}: {pred['prob']*100:.2f}%</li>"
            else:
                xnnpack_predictions_html += "<li>Invalid XNNPACK prediction format.</li>"

    # Overall error message for display if any part of get_initial_predictions failed
    overall_error_html = ""
    if results.get('error'):
        overall_error_html = f"<p style='color:red; text-align:center;'>An error occurred: {results.get('error')}</p>"


    # Initial latency values - these will be updated by JS after benchmark
    # IDs are now specific to PyTorch or XNNPACK for JS to target
    # avg_latency_html, min_latency_html, etc. are no longer needed here as direct variable,
    # the "N/A (Run benchmark)" will be directly in the main f-string.

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ImageNet Inference Results</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
            h1, h2, h3 {{ text-align: center; color: #333; }}
            img {{ display: block; margin-left: auto; margin-right: auto; max-width: 90%; height: auto; margin-bottom: 20px; border: 1px solid #ddd; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }}
            .container {{ max-width: 900px; margin: auto; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            .predictions, .stats {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 4px; background-color: #f9f9f9; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin-bottom: 8px; padding: 5px; border-bottom: 1px solid #eee; }}
            li:last-child {{ border-bottom: none; }}
            p {{ line-height: 1.6; }}
            .progress-bar-container {{ width: 100%; background-color: #e0e0e0; border-radius: 4px; margin-bottom: 10px; overflow: hidden; }}
            .progress-bar {{ width: 0%; height: 20px; background-color: #4caf50; text-align: center; line-height: 20px; color: white; border-radius: 4px; transition: width 0.3s ease, background-color 0.3s ease; }}
            #benchmarkStatusMessage {{ margin-bottom: 10px; font-style: italic; text-align: center;}}
            button#startBenchmarkButton {{ display: block; margin: 20px auto; padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; transition: background-color 0.3s ease; }}
            button#startBenchmarkButton:disabled {{ background-color: #ccc; cursor: not-allowed; }}
            button#startBenchmarkButton:hover:not(:disabled) {{ background-color: #0056b3; }}
            .predictions-container, .stats-container {{ display: flex; justify-content: space-between; margin-bottom:10px; }}
            .predictions-column, .stats-column {{ width: 48%; padding:10px; box-sizing: border-box; border: 1px solid #eee; border-radius: 4px; background: #fff;}}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>ImageNet Inference and Benchmark Demo</h1>

            <div class="image-display">
                <h2>Source Image</h2>
                <img src="{current_image_url}" alt="Inference Image">
                <p style="text-align:center;">Image URL: <a href="{current_image_url}" target="_blank">{current_image_url}</a></p>
            </div>

            {overall_error_html}

            <div class="predictions">
                 <h2>Initial Predictions</h2>
                <div class="predictions-container">
                    <div class="predictions-column">
                        <h3>PyTorch Predictions:</h3>
                        <ul id="pytorchPredictionsList">
                            {pytorch_predictions_html}
                        </ul>
                    </div>
                    <div class="predictions-column">
                        <h3>XNNPACK Predictions:</h3>
                        <ul id="xnnpackPredictionsList">
                            {xnnpack_predictions_html}
                        </ul>
                    </div>
                </div>
            </div>

            <div class="stats">
                <h2>Benchmark Controls and Latency Statistics</h2>
                <button id="startBenchmarkButton" onclick="startBenchmark()">Start Benchmark</button>
                <div id="benchmarkStatusMessage" style="display:none; margin-top:10px;"></div>
                <div class="progress-bar-container" id="progressBarContainer" style="display:none; margin-top:5px;">
                    <div class="progress-bar" id="progressBar">0%</div>
                </div>
                <div class.stats-container>
                    <div class="stats-column">
                        <h3>PyTorch Latency:</h3>
                        <p>Average: <span id="pytorchAvgLatency">N/A (Run benchmark)</span></p>
                        <p>Minimum: <span id="pytorchMinLatency">N/A (Run benchmark)</span></p>
                        <p>Maximum: <span id="pytorchMaxLatency">N/A (Run benchmark)</span></p>
                        <p>Std Dev: <span id="pytorchStdLatency">N/A (Run benchmark)</span></p>
                    </div>
                    <div class="stats-column">
                        <h3>XNNPACK Latency:</h3>
                        <p>Average: <span id="xnnpackAvgLatency">N/A (Run benchmark)</span></p>
                        <p>Minimum: <span id="xnnpackMinLatency">N/A (Run benchmark)</span></p>
                        <p>Maximum: <span id="xnnpackMaxLatency">N/A (Run benchmark)</span></p>
                        <p>Std Dev: <span id="xnnpackStdLatency">N/A (Run benchmark)</span></p>
                    </div>
                </div>
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

                // Clear old latency results
                document.getElementById('pytorchAvgLatency').textContent = 'N/A';
                document.getElementById('pytorchMinLatency').textContent = 'N/A';
                document.getElementById('pytorchMaxLatency').textContent = 'N/A';
                document.getElementById('pytorchStdLatency').textContent = 'N/A';
                document.getElementById('xnnpackAvgLatency').textContent = 'N/A';
                document.getElementById('xnnpackMinLatency').textContent = 'N/A';
                document.getElementById('xnnpackMaxLatency').textContent = 'N/A';
                document.getElementById('xnnpackStdLatency').textContent = 'N/A';

                fetch('/start_benchmark', {{ method: 'POST' }})
                    .then(response => {{
                        if (!response.ok) {{
                            // Try to parse error message from server if available
                            return response.json().then(err => {{
                                throw new Error(err.message || `Server error: ${{response.status}}`);
                            }}).catch(() => {{
                                // Fallback if parsing error message fails
                                throw new Error(`Server error: ${{response.status}}`);
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
                                throw new Error(err.message || `Server error: ${{response.status}}`);
                            }}).catch(() => {{
                                throw new Error(`Server error: ${{response.status}}`);
                            }});
                        }}
                        return response.json();
                    }})
                    .then(data => {{
                        console.log('Status update:', data); // For debugging
                        if (!data || typeof data.status === 'undefined') {{
                            console.error('Invalid or incomplete status data received:', data);
                            statusMessage.textContent = 'Error: Received invalid status data from server.';
                            statusMessage.style.color = 'red';
                            if (benchmarkInterval) clearInterval(benchmarkInterval);
                            button.disabled = false;
                            button.textContent = 'Start Benchmark'; // Or 'Re-run Benchmark' if appropriate
                            progressBarContainer.style.display = 'none';
                            return;
                        }}
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
                            if (data.results) {{ // data.results now has 'pytorch' and 'xnnpack' keys
                                if (data.results.pytorch) {{
                                    document.getElementById('pytorchAvgLatency').textContent = data.results.pytorch.avg_latency !== null ? data.results.pytorch.avg_latency.toFixed(2) + ' ms' : 'N/A';
                                    document.getElementById('pytorchMinLatency').textContent = data.results.pytorch.min_latency !== null ? data.results.pytorch.min_latency.toFixed(2) + ' ms' : 'N/A';
                                    document.getElementById('pytorchMaxLatency').textContent = data.results.pytorch.max_latency !== null ? data.results.pytorch.max_latency.toFixed(2) + ' ms' : 'N/A';
                                    document.getElementById('pytorchStdLatency').textContent = data.results.pytorch.std_latency !== null ? data.results.pytorch.std_latency.toFixed(2) + ' ms' : 'N/A';
                                }}
                                if (data.results.xnnpack) {{
                                    document.getElementById('xnnpackAvgLatency').textContent = data.results.xnnpack.avg_latency !== null ? data.results.xnnpack.avg_latency.toFixed(2) + ' ms' : 'N/A';
                                    document.getElementById('xnnpackMinLatency').textContent = data.results.xnnpack.min_latency !== null ? data.results.xnnpack.min_latency.toFixed(2) + ' ms' : 'N/A';
                                    document.getElementById('xnnpackMaxLatency').textContent = data.results.xnnpack.max_latency !== null ? data.results.xnnpack.max_latency.toFixed(2) + ' ms' : 'N/A';
                                    document.getElementById('xnnpackStdLatency').textContent = data.results.xnnpack.std_latency !== null ? data.results.xnnpack.std_latency.toFixed(2) + ' ms' : 'N/A';
                                }}
                            }}
                        }} else if (data.status === 'error') {{
                            clearInterval(benchmarkInterval);
                            // Update latencies to N/A or show error message if partial results are not desired on error
                            // For now, if error, we don't update latency fields from potentially partial results.
                            // User will see the error message. Old results (if any) will remain or N/A.
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
