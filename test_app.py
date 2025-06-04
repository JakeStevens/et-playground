import unittest
from unittest.mock import patch, MagicMock
import time # Add this
# import threading # Not strictly needed for these tests

from flask import Flask
from PIL import Image as PILImage # Renamed to avoid conflict if Image is used elsewhere directly
import torch
import requests # For requests.exceptions.RequestException
from io import BytesIO

# Assuming app.py and constants.py are in the same directory or accessible via PYTHONPATH
from app import app # Keep this for app.test_client() and app.app_context()
import app as main_app # To access module-level globals and functions from app.py
from constants import IMAGENET_CLASSES, IMAGE_URL # IMAGE_URL still used in some original tests

class TestApp(unittest.TestCase):

    def setUp(self):
        app.testing = True # app here refers to the Flask app instance from 'from app import app'
        self.client = app.test_client()
        # Reset global state from app.py before each test
        main_app.benchmark_status = { # Use main_app to access module-level globals
            'status': 'idle', 'progress': 0, 'results': None, 'error_message': None
        }
        main_app._global_model_for_benchmark = None
        main_app._global_tensor_for_benchmark = None

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

    @patch('app.get_top5_predictions') # Patches 'app.get_top5_predictions'
    @patch('app.load_model')      # Patches 'app.load_model'
    @patch('app.load_and_preprocess_image') # Patches 'app.load_and_preprocess_image'
    def test_get_inference_results_modified(self, mock_load_image, mock_load_model, mock_get_preds):
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_load_image.return_value = mock_tensor

        mock_model_instance = MagicMock()
        mock_load_model.return_value = mock_model_instance

        mock_predictions = [{'name': 'mock_cat', 'prob': 0.9}]
        mock_get_preds.return_value = mock_predictions

        # Call the actual function from main_app module
        results = main_app.get_inference_results()

        self.assertIsNone(results['error'])
        self.assertEqual(results['top_5_predictions'], mock_predictions)
        self.assertEqual(results['image_url'], main_app.IMAGE_URL) # Access IMAGE_URL via main_app

        self.assertNotIn('avg_latency', results)
        self.assertNotIn('min_latency', results)

        mock_load_image.assert_called_once()
        mock_load_model.assert_called_once()

        self.assertEqual(main_app._global_model_for_benchmark, mock_model_instance)
        self.assertEqual(main_app._global_tensor_for_benchmark, mock_tensor)

        self.assertEqual(mock_model_instance.call_count, 5) # Warm-up calls
        mock_get_preds.assert_called_once_with(mock_model_instance, mock_tensor, main_app.IMAGENET_CLASSES)

    @patch('app.load_model')
    @patch('app.load_and_preprocess_image')
    def test_get_inference_results_image_load_failure(self, mock_load_image, mock_load_model):
        """Test get_inference_results when image loading fails."""
        mock_load_image.return_value = None
        mock_load_model.return_value = MagicMock() # Model loading succeeds

        results = main_app.get_inference_results() # Use main_app

        self.assertIsNotNone(results['error'])
        self.assertIn("Failed to load and preprocess image", results['error'])
        self.assertEqual(results['top_5_predictions'], [])
        # Latency fields are no longer in results dict
        self.assertNotIn('avg_latency', results)

    @patch('app.load_model')
    @patch('app.load_and_preprocess_image')
    def test_get_inference_results_model_load_failure(self, mock_load_image, mock_load_model):
        """Test get_inference_results when model loading fails."""
        mock_load_image.return_value = MagicMock(spec=torch.Tensor) # Image loading succeeds
        mock_load_model.return_value = None

        results = main_app.get_inference_results() # Use main_app

        self.assertIsNotNone(results['error'])
        self.assertIn("Failed to load model", results['error'])
        self.assertEqual(results['top_5_predictions'], [])
        self.assertNotIn('avg_latency', results)

    @patch('app.load_model')
    @patch('app.load_and_preprocess_image')
    def test_get_inference_results_both_load_failure(self, mock_load_image, mock_load_model):
        """Test get_inference_results when both image and model loading fail."""
        mock_load_image.return_value = None
        mock_load_model.return_value = None

        results = main_app.get_inference_results() # Use main_app

        self.assertIsNotNone(results['error'])
        self.assertIn("Failed to load model and image", results['error']) # Check for specific combined message
        self.assertEqual(results['top_5_predictions'], [])
        self.assertNotIn('avg_latency', results)

    # New tests:
    def test_start_benchmark_route_success(self):
        # Mock necessary components to avoid actual model loading/computation
        with patch('app.load_model', return_value=MagicMock()), \
             patch('app.load_and_preprocess_image', return_value=MagicMock()), \
             patch('app._run_benchmark_thread') as mock_run_benchmark: # Patching 'app._run_benchmark_thread'

            # Initial call to '/' to populate _global_model_for_benchmark etc.
            # This relies on get_inference_results being called by '/' and populating globals
            self.client.get('/')

            response = self.client.post('/start_benchmark')
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data['message'], 'Benchmark started.')

            self.assertEqual(main_app.benchmark_status['status'], 'running') # Check module global
            self.assertEqual(main_app.benchmark_status['progress'], 0)

            mock_run_benchmark.assert_called_once()

    def test_start_benchmark_route_already_running(self):
        main_app.benchmark_status['status'] = 'running' # Simulate running state

        response = self.client.post('/start_benchmark')
        self.assertEqual(response.status_code, 409) # Conflict
        data = response.get_json()
        self.assertEqual(data['message'], 'Benchmark is already running.')

    def test_benchmark_status_route(self):
        main_app.benchmark_status = {'status': 'running', 'progress': 50, 'results': None, 'error_message': None}

        response = self.client.get('/benchmark_status')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'running')
        self.assertEqual(data['progress'], 50)

    @patch('app.benchmark_inference')
    def test_run_benchmark_thread_success(self, mock_benchmark_inference_actual):
        mock_model = MagicMock()
        mock_tensor = MagicMock()
        mock_benchmark_inference_actual.return_value = (10.0, 5.0, 15.0, 1.0)

        main_app._global_model_for_benchmark = mock_model
        main_app._global_tensor_for_benchmark = mock_tensor
        main_app.benchmark_status = {'status': 'running', 'progress': 0, 'results': None, 'error_message': None}

        main_app._run_benchmark_thread()

        self.assertEqual(main_app.benchmark_status['status'], 'complete')
        self.assertIsNotNone(main_app.benchmark_status['results'])
        self.assertEqual(main_app.benchmark_status['results']['avg_latency'], 10.0)
        mock_benchmark_inference_actual.assert_called_once_with(mock_model, mock_tensor)

    @patch('app.benchmark_inference', side_effect=Exception("Test benchmark error"))
    def test_run_benchmark_thread_error(self, mock_benchmark_inference_error):
        mock_model = MagicMock()
        mock_tensor = MagicMock()

        main_app._global_model_for_benchmark = mock_model
        main_app._global_tensor_for_benchmark = mock_tensor
        main_app.benchmark_status = {'status': 'running', 'progress': 0, 'results': None, 'error_message': None}

        main_app._run_benchmark_thread()

        self.assertEqual(main_app.benchmark_status['status'], 'error')
        self.assertEqual(main_app.benchmark_status['error_message'], "Test benchmark error")
        mock_benchmark_inference_error.assert_called_once_with(mock_model, mock_tensor)

    def test_run_benchmark_thread_no_model_tensor(self):
        main_app._global_model_for_benchmark = None
        main_app._global_tensor_for_benchmark = None
        main_app.benchmark_status = {'status': 'idle', 'progress': 0, 'results': None, 'error_message': None}

        main_app._run_benchmark_thread()

        self.assertEqual(main_app.benchmark_status['status'], 'error')
        self.assertEqual(main_app.benchmark_status['error_message'], 'Model or tensor not loaded for benchmark.')

    # Updated test_index_route (original success one)
    @patch('app.get_inference_results')
    def test_index_route_initial_load_modified(self, mock_get_inference_results_call):
        mock_initial_results = {
            'top_5_predictions': [{'name': 'TestCat', 'prob': 0.9876}],
            'image_url': 'http://example.com/test_image.jpg',
            'error': None
        }
        mock_get_inference_results_call.return_value = mock_initial_results

        response = self.client.get('/')
        response_data = response.data.decode('utf-8')

        self.assertEqual(response.status_code, 200)
        mock_get_inference_results_call.assert_called_once()

        self.assertIn('TestCat', response_data)
        self.assertIn('98.7600%', response_data)
        self.assertIn('http://example.com/test_image.jpg', response_data)

        self.assertIn('<button id="startBenchmarkButton"', response_data)
        self.assertIn("N/A (Run benchmark)", response_data)
        self.assertIn('id="avgLatency"', response_data)
        self.assertNotIn('123.45 ms', response_data) # Old direct latency

    # Update test_index_route_with_error_from_get_inference_results
    @patch('app.get_inference_results')
    def test_index_route_with_error_from_get_inference_results_modified(self, mock_get_results):
        error_message = 'Simulated error during inference'
        mock_data = {
            'top_5_predictions': [],
            'image_url': main_app.IMAGE_URL,
            'error': error_message
        }
        mock_get_results.return_value = mock_data

        response = self.client.get('/')
        response_data = response.data.decode('utf-8')

        self.assertEqual(response.status_code, 200)
        mock_get_results.assert_called_once()

        self.assertIn(f"Error fetching initial predictions: {error_message}", response_data)

        self.assertIn("Average Latency: <span id=\"avgLatency\">N/A (Run benchmark)</span>", response_data)
        self.assertIn("Minimum Latency: <span id=\"minLatency\">N/A (Run benchmark)</span>", response_data)

        self.assertIn(main_app.IMAGE_URL, response_data)

if __name__ == '__main__':
    unittest.main()
