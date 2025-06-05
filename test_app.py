import unittest
from unittest.mock import patch, MagicMock
import time
import threading # Added for testing thread creation

from flask import Flask
from PIL import Image as PILImage
import torch
import requests # For requests.exceptions.RequestException
from io import BytesIO

# Assuming app.py and constants.py are in the same directory or accessible via PYTHONPATH
from app import app # Flask app instance
import app as main_app # The app module itself
from constants import IMAGENET_CLASSES, IMAGE_URL

class TestApp(unittest.TestCase):

    def setUp(self):
        app.testing = True # app is the Flask app instance
        self.client = app.test_client()
        # Create a new manager instance for each test to ensure isolation
        main_app.benchmark_manager = main_app.BenchmarkManager()
        # It's also good practice to ensure the manager's internal model/tensor are reset
        main_app.benchmark_manager.model = None
        main_app.benchmark_manager.image_tensor = None

    def test_example(self):
        """A placeholder test to ensure the structure is valid."""
        self.assertTrue(True)

    @patch('app.requests.get')
    def test_load_and_preprocess_image_success(self, mock_get):
        """Test successful image loading and preprocessing."""
        # Configure the mock response for requests.get
        mock_image = PILImage.new('RGB', (256, 256), color='blue') # Create a dummy image
        mock_image_bytes = BytesIO()
        mock_image.save(mock_image_bytes, format='JPEG')
        mock_image_bytes.seek(0)  # Reset buffer position to the beginning

        mock_response = MagicMock()
        mock_response.raw = mock_image_bytes
        mock_get.return_value = mock_response

        # Call the function
        tensor = load_and_preprocess_image()

        # Assertions
        self.assertIsNotNone(tensor)
        self.assertIsInstance(tensor, torch.Tensor)
        # Expected shape: (batch_size, channels, height, width)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    @patch('app.requests.get')
    def test_load_and_preprocess_image_download_failure(self, mock_get):
        """Test image loading failure due to a download error."""
        # Configure the mock to raise an exception
        mock_get.side_effect = requests.exceptions.RequestException

        # Call the function
        result = load_and_preprocess_image()

        # Assertions
        self.assertIsNone(result)

    @patch('app.models.resnet18')
    def test_load_model_success(self, mock_resnet18):
        """Test successful model loading."""
        # Configure the mock model
        mock_model_instance = MagicMock()
        mock_resnet18.return_value = mock_model_instance

        # Call the function
        model = load_model()

        # Assertions
        self.assertIsNotNone(model)
        self.assertEqual(model, mock_model_instance)
        mock_resnet18.assert_called_once_with(pretrained=True)
        mock_model_instance.to.assert_called_once_with('cpu')
        mock_model_instance.eval.assert_called_once()

    @patch('app.models.resnet18')
    def test_load_model_failure(self, mock_resnet18):
        """Test model loading failure."""
        # Configure the mock to raise an exception
        mock_resnet18.side_effect = Exception("Model loading error")

        # Call the function
        model = load_model()

        # Assertions
        self.assertIsNone(model)

    # Tests for standalone utility functions (load_model, load_and_preprocess_image)
    # These are largely unchanged as the functions they test were not removed from app.py

    # --- Tests for BenchmarkManager ---
    def test_bm_initial_state(self):
        manager = main_app.BenchmarkManager()
        self.assertEqual(manager.status, 'idle')
        self.assertEqual(manager.progress, 0)
        self.assertIsNone(manager.results)
        self.assertIsNone(manager.error_message)
        self.assertIsNone(manager.model)
        self.assertIsNone(manager.image_tensor)

    @patch('app.load_model')
    @patch('app.load_and_preprocess_image')
    def test_bm_load_resources_if_needed_success(self, mock_load_image, mock_load_model):
        mock_model_instance = MagicMock()
        mock_load_model.return_value = mock_model_instance
        mock_tensor_instance = MagicMock()
        mock_load_image.return_value = mock_tensor_instance

        manager = main_app.BenchmarkManager()
        loaded = manager.load_resources_if_needed()

        self.assertTrue(loaded)
        self.assertEqual(manager.status, 'idle') # Should be idle after loading and warmup
        self.assertEqual(manager.model, mock_model_instance)
        self.assertEqual(manager.image_tensor, mock_tensor_instance)
        mock_load_model.assert_called_once()
        mock_load_image.assert_called_once()
        self.assertEqual(mock_model_instance.call_count, 5) # 5 warm-up calls

    @patch('app.load_model', return_value=None)
    @patch('app.load_and_preprocess_image')
    def test_bm_load_resources_if_needed_model_fail(self, mock_load_image, mock_load_model):
        mock_load_image.return_value = MagicMock() # Image load succeeds
        manager = main_app.BenchmarkManager()
        loaded = manager.load_resources_if_needed()

        self.assertFalse(loaded)
        self.assertEqual(manager.status, 'error')
        self.assertIn("Model loading failed", manager.error_message)
        self.assertIsNone(manager.model)

    @patch('app.load_model')
    @patch('app.load_and_preprocess_image', return_value=None)
    def test_bm_load_resources_if_needed_image_fail(self, mock_load_image, mock_load_model):
        mock_load_model.return_value = MagicMock() # Model load succeeds
        manager = main_app.BenchmarkManager()
        loaded = manager.load_resources_if_needed()

        self.assertFalse(loaded)
        self.assertEqual(manager.status, 'error')
        self.assertIn("Image tensor loading failed", manager.error_message)
        self.assertIsNone(manager.image_tensor)

    @patch('app.load_model')
    @patch('app.load_and_preprocess_image')
    def test_bm_load_resources_if_needed_already_loaded(self, mock_load_image, mock_load_model):
        manager = main_app.BenchmarkManager()
        manager.model = MagicMock()
        manager.image_tensor = MagicMock()

        loaded = manager.load_resources_if_needed()
        self.assertTrue(loaded)
        mock_load_image.assert_not_called()
        mock_load_model.assert_not_called()

    @patch('app.get_top5_predictions')
    def test_bm_get_initial_predictions_success(self, mock_get_top5):
        manager = main_app.BenchmarkManager()
        manager.model = MagicMock() # Assume loaded
        manager.image_tensor = MagicMock() # Assume loaded
        mock_preds = [{'name': 'cat', 'prob': 0.9}]
        mock_get_top5.return_value = mock_preds

        with patch.object(manager, 'load_resources_if_needed', return_value=True):
            results = manager.get_initial_predictions()

        self.assertIsNone(results['error'])
        self.assertEqual(results['top_5_predictions'], mock_preds)
        mock_get_top5.assert_called_once_with(manager.model, manager.image_tensor, main_app.IMAGENET_CLASSES)

    def test_bm_get_initial_predictions_load_fail(self):
        manager = main_app.BenchmarkManager()
        with patch.object(manager, 'load_resources_if_needed', return_value=False) as mock_load:
            manager.error_message = "Resource load failed" # Simulate error set by load_resources
            results = manager.get_initial_predictions()

        mock_load.assert_called_once()
        self.assertIsNotNone(results['error'])
        self.assertEqual(results['error'], "Resource load failed")
        self.assertEqual(results['top_5_predictions'], [])

    @patch('threading.Thread')
    def test_bm_start_benchmark_success(self, mock_thread_constructor):
        manager = main_app.BenchmarkManager()
        manager.model = MagicMock() # Pre-load
        manager.image_tensor = MagicMock() # Pre-load
        mock_thread_instance = MagicMock()
        mock_thread_constructor.return_value = mock_thread_instance

        with patch.object(manager, 'load_resources_if_needed', return_value=True):
            message, success = manager.start_benchmark()

        self.assertTrue(success)
        self.assertEqual(message['message'], 'Benchmark started.')
        self.assertEqual(manager.status, 'running')
        self.assertEqual(manager.progress, 0)
        self.assertIsNone(manager.results)
        self.assertIsNone(manager.error_message)
        mock_thread_constructor.assert_called_once_with(target=manager._perform_benchmark_and_update_status)
        mock_thread_instance.start.assert_called_once()
        self.assertTrue(mock_thread_instance.daemon)


    def test_bm_start_benchmark_already_active(self):
        manager = main_app.BenchmarkManager()
        manager.status = 'running'
        message, success = manager.start_benchmark()
        self.assertFalse(success)
        self.assertIn('already active', message['message'])

        manager.status = 'loading'
        message, success = manager.start_benchmark()
        self.assertFalse(success)
        self.assertIn('already active', message['message'])

    def test_bm_start_benchmark_load_fail(self):
        manager = main_app.BenchmarkManager()
        with patch.object(manager, 'load_resources_if_needed', return_value=False) as mock_load:
            manager.error_message = "Load error" # Simulate error from loading
            message, success = manager.start_benchmark()

        self.assertFalse(success)
        self.assertEqual(message['message'], 'Failed to load resources for benchmarking.')
        self.assertEqual(message['error'], 'Load error')
        mock_load.assert_called_once()

    @patch('app.benchmark_inference') # Patch the global benchmark_inference in app.py
    def test_bm_perform_benchmark_and_update_status_success(self, mock_bi_func):
        manager = main_app.BenchmarkManager()
        manager.model = MagicMock()
        manager.image_tensor = MagicMock()
        latency_data = {'avg_latency': 10.0, 'min_latency': 5.0}
        mock_bi_func.return_value = latency_data

        manager._perform_benchmark_and_update_status()

        self.assertEqual(manager.status, 'complete')
        self.assertEqual(manager.results, latency_data)
        self.assertEqual(manager.progress, 100)
        mock_bi_func.assert_called_once_with(manager.model, manager.image_tensor, manager._update_progress_callback)

    @patch('app.benchmark_inference', side_effect=Exception("Benchmark func error"))
    def test_bm_perform_benchmark_and_update_status_exception(self, mock_bi_func_exception):
        manager = main_app.BenchmarkManager()
        manager.model = MagicMock()
        manager.image_tensor = MagicMock()

        manager._perform_benchmark_and_update_status()

        self.assertEqual(manager.status, 'error')
        self.assertEqual(manager.error_message, "Benchmark func error")
        self.assertIsNone(manager.results)
        mock_bi_func_exception.assert_called_once_with(manager.model, manager.image_tensor, manager._update_progress_callback)

    def test_bm_update_progress_callback(self):
        manager = main_app.BenchmarkManager()
        manager.status = 'running' # Must be running to accept progress
        manager._update_progress_callback(50)
        self.assertEqual(manager.progress, 50)

        manager.status = 'complete' # Should not update if not running
        manager._update_progress_callback(75)
        self.assertEqual(manager.progress, 50) # Progress should remain 50

    # --- Test for refactored benchmark_inference ---
    def test_benchmark_inference_logic(self):
        mock_model = MagicMock()
        mock_tensor = MagicMock()
        mock_progress_callback = MagicMock()

        results = main_app.benchmark_inference(mock_model, mock_tensor, mock_progress_callback)

        self.assertIn('avg_latency', results)
        self.assertIsInstance(results['avg_latency'], float)
        # Check that model was called 1000 times
        self.assertEqual(mock_model.call_count, 1000)
        # Check progress callback was called, e.g., last call should be 100
        mock_progress_callback.assert_called_with(100)
        self.assertGreaterEqual(mock_progress_callback.call_count, 1000) # Called for each iteration

        with self.assertRaises(ValueError):
            main_app.benchmark_inference(None, mock_tensor, mock_progress_callback)
        with self.assertRaises(ValueError):
            main_app.benchmark_inference(mock_model, None, mock_progress_callback)


    # --- Route Tests (Refactored) ---
    @patch('app.main_app.benchmark_manager.get_initial_predictions')
    def test_index_route_initial_load_modified(self, mock_get_initial_predictions):
        mock_data = {
            'top_5_predictions': [{'name': 'TestCat', 'prob': 0.9876}],
            'image_url': 'http://example.com/test_image.jpg',
            'error': None
        }
        mock_get_initial_predictions.return_value = mock_data

        response = self.client.get('/')
        response_data = response.data.decode('utf-8')

        self.assertEqual(response.status_code, 200)
        mock_get_initial_predictions.assert_called_once()
        self.assertIn('TestCat', response_data)
        self.assertIn('98.7600%', response_data)
        self.assertIn('http://example.com/test_image.jpg', response_data)
        self.assertIn('<button id="startBenchmarkButton"', response_data)
        self.assertIn("N/A (Run benchmark)", response_data)

    @patch('app.main_app.benchmark_manager.get_initial_predictions')
    def test_index_route_with_error_from_get_inference_results_modified(self, mock_get_initial_predictions):
        error_message = 'Simulated error during initial predictions'
        mock_data = {
            'top_5_predictions': [],
            'image_url': main_app.IMAGE_URL,
            'error': error_message
        }
        mock_get_initial_predictions.return_value = mock_data

        response = self.client.get('/')
        response_data = response.data.decode('utf-8')

        self.assertEqual(response.status_code, 200)
        mock_get_initial_predictions.assert_called_once()
        self.assertIn(f"Error fetching initial predictions: {error_message}", response_data)
        self.assertIn("Average Latency: <span id=\"avgLatency\">N/A (Run benchmark)</span>", response_data)

    @patch('app.main_app.benchmark_manager.start_benchmark')
    def test_start_benchmark_route_success(self, mock_start_benchmark):
        mock_start_benchmark.return_value = ({'message': 'Benchmark started.'}, True)
        response = self.client.post('/start_benchmark')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['message'], 'Benchmark started.')
        mock_start_benchmark.assert_called_once()

    @patch('app.main_app.benchmark_manager.start_benchmark')
    def test_start_benchmark_route_already_running(self, mock_start_benchmark):
        mock_start_benchmark.return_value = ({'message': 'Benchmark process is already active (running). Please wait.'}, False)
        # Simulate manager being in 'running' state for status code decision in route
        with patch.object(main_app.benchmark_manager, 'get_status', return_value={'status': 'running'}):
            response = self.client.post('/start_benchmark')
        self.assertEqual(response.status_code, 409)
        data = response.get_json()
        self.assertIn('already active', data['message'])
        mock_start_benchmark.assert_called_once()

    @patch('app.main_app.benchmark_manager.get_status')
    def test_benchmark_status_route(self, mock_get_status):
        mock_status_data = {'status': 'running', 'progress': 50, 'results': None, 'error_message': None}
        mock_get_status.return_value = mock_status_data

        response = self.client.get('/benchmark_status')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data, mock_status_data)
        mock_get_status.assert_called_once()

if __name__ == '__main__':
    unittest.main()
