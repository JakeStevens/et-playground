import unittest
from unittest.mock import patch, MagicMock

from flask import Flask
from PIL import Image as PILImage # Renamed to avoid conflict if Image is used elsewhere directly
import torch
import requests # For requests.exceptions.RequestException
from io import BytesIO

# Assuming app.py and constants.py are in the same directory or accessible via PYTHONPATH
from app import app, load_and_preprocess_image, load_model, get_inference_results
from constants import IMAGENET_CLASSES, IMAGE_URL

class TestApp(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.client = app.test_client()

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

    @patch('app.get_top5_predictions')
    @patch('app.benchmark_inference')
    @patch('app.load_model')
    @patch('app.load_and_preprocess_image')
    def test_get_inference_results_success(self, mock_load_image, mock_load_model, mock_benchmark, mock_get_preds):
        """Test successful run of get_inference_results."""
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_load_image.return_value = mock_tensor

        mock_model_instance = MagicMock()
        mock_load_model.return_value = mock_model_instance

        mock_latencies = (10.0, 5.0, 15.0, 1.0)
        mock_benchmark.return_value = mock_latencies

        mock_predictions = [{'name': 'mock_cat', 'prob': 0.9}]
        mock_get_preds.return_value = mock_predictions

        results = get_inference_results()

        self.assertIsNone(results['error'])
        self.assertEqual(results['top_5_predictions'], mock_predictions)
        self.assertEqual(results['avg_latency'], mock_latencies[0])
        self.assertEqual(results['min_latency'], mock_latencies[1])
        self.assertEqual(results['max_latency'], mock_latencies[2])
        self.assertEqual(results['std_latency'], mock_latencies[3])
        self.assertEqual(results['image_url'], IMAGE_URL)

        mock_load_image.assert_called_once()
        mock_load_model.assert_called_once()

        # Check warm-up calls: model(tensor) is called 5 times
        # MagicMock records calls to itself.
        # The model is also called by benchmark_inference and get_top5_predictions.
        # We are primarily interested in the 5 warm-up calls.
        # mock_model_instance.assert_any_call(mock_tensor) # Check it was called with the tensor

        # Count calls to the model instance with the mock_tensor.
        # The first 5 calls should be the warm-up calls.
        # Then benchmark_inference calls it 1000 times.
        # Then get_top5_predictions calls it once.
        # So, total calls to model_instance(mock_tensor) = 5 (warmup) + 1000 (benchmark) + 1 (preds)
        # For simplicity in this unit test, we'll check the direct helper calls.
        # Verifying the exact number of warm-up calls on the mock_model_instance
        # can be tricky if other parts of the code also call it.
        # However, the prompt asks for it.
        # The model itself is called, so mock_model_instance is the callable.

        # Check that the model was called with the tensor at least 5 times for warm-up
        # This is a bit indirect. A more direct way would be to check call_args_list
        # call_list = mock_model_instance.call_args_list
        # warmup_calls = [call for call in call_list if call == unittest.mock.call(mock_tensor)]
        # self.assertGreaterEqual(len(warmup_calls), 5)
        # For now, let's check the count of calls to the mocked model object itself.
        # The model is called 5 times in warmup, 1000 times in benchmark, 1 time in get_top5
        # Total expected calls = 5 (warmup) + 1000 (benchmark_inference internal) + 1 (get_top5_predictions internal)
        # The mock_model_instance is the model itself.
        # The benchmark_inference and get_top5_predictions are separate mocks here.
        # So we only expect the 5 warm-up calls on the *mock_model_instance* directly from get_inference_results.
        self.assertEqual(mock_model_instance.call_count, 5, "Model should be called 5 times for warm-up.")


        mock_benchmark.assert_called_once_with(mock_model_instance, mock_tensor)
        mock_get_preds.assert_called_once_with(mock_model_instance, mock_tensor, IMAGENET_CLASSES)


    @patch('app.load_model')
    @patch('app.load_and_preprocess_image')
    def test_get_inference_results_image_load_failure(self, mock_load_image, mock_load_model):
        """Test get_inference_results when image loading fails."""
        mock_load_image.return_value = None
        mock_load_model.return_value = MagicMock() # Model loading succeeds

        results = get_inference_results()

        self.assertIsNotNone(results['error'])
        self.assertIn("Failed to load and preprocess image", results['error'])
        self.assertEqual(results['top_5_predictions'], [])
        self.assertIsNone(results['avg_latency'])
        self.assertIsNone(results['min_latency'])
        self.assertIsNone(results['max_latency'])
        self.assertIsNone(results['std_latency'])

    @patch('app.load_model')
    @patch('app.load_and_preprocess_image')
    def test_get_inference_results_model_load_failure(self, mock_load_image, mock_load_model):
        """Test get_inference_results when model loading fails."""
        mock_load_image.return_value = MagicMock(spec=torch.Tensor) # Image loading succeeds
        mock_load_model.return_value = None

        results = get_inference_results()

        self.assertIsNotNone(results['error'])
        self.assertIn("Failed to load model", results['error'])
        self.assertEqual(results['top_5_predictions'], [])
        self.assertIsNone(results['avg_latency'])
        self.assertIsNone(results['min_latency'])
        self.assertIsNone(results['max_latency'])
        self.assertIsNone(results['std_latency'])

    @patch('app.load_model')
    @patch('app.load_and_preprocess_image')
    def test_get_inference_results_both_load_failure(self, mock_load_image, mock_load_model):
        """Test get_inference_results when both image and model loading fail."""
        mock_load_image.return_value = None
        mock_load_model.return_value = None

        results = get_inference_results()

        self.assertIsNotNone(results['error'])
        self.assertIn("Failed to load model and image", results['error']) # Check for specific combined message
        self.assertEqual(results['top_5_predictions'], [])
        self.assertIsNone(results['avg_latency'])
        self.assertIsNone(results['min_latency'])
        self.assertIsNone(results['max_latency'])
        self.assertIsNone(results['std_latency'])

    @patch('app.get_inference_results')
    def test_index_route_success(self, mock_get_results):
        """Test the index route with a successful inference result."""
        mock_data = {
            'top_5_predictions': [{'name': 'TestCat', 'prob': 0.9876}],
            'avg_latency': 123.45,
            'min_latency': 100.0,
            'max_latency': 150.0,
            'std_latency': 10.0,
            'image_url': 'http://example.com/test_image.jpg',
            'error': None
        }
        mock_get_results.return_value = mock_data

        response = self.client.get('/')
        response_data = response.data.decode('utf-8')

        self.assertEqual(response.status_code, 200)
        mock_get_results.assert_called_once() # Ensure get_inference_results was called

        self.assertIn('TestCat', response_data)
        self.assertIn('98.7600%', response_data) # Check formatted probability
        self.assertIn('123.45 ms', response_data)
        self.assertIn('100.00 ms', response_data) # min_latency
        self.assertIn('150.00 ms', response_data) # max_latency
        self.assertIn('10.00 ms', response_data) # std_latency
        self.assertIn('http://example.com/test_image.jpg', response_data)
        self.assertNotIn('Error:', response_data) # Error message should not be present

    @patch('app.get_inference_results')
    def test_index_route_with_error_from_get_inference_results(self, mock_get_results):
        """Test the index route when get_inference_results returns an error."""
        error_message = 'Simulated error during inference'
        mock_data = {
            'top_5_predictions': [],
            'avg_latency': None,
            'min_latency': None,
            'max_latency': None,
            'std_latency': None,
            'image_url': IMAGE_URL, # Use the actual constant for consistency
            'error': error_message
        }
        mock_get_results.return_value = mock_data

        response = self.client.get('/')
        response_data = response.data.decode('utf-8')

        self.assertEqual(response.status_code, 200)
        mock_get_results.assert_called_once()

        self.assertIn(f"Error: {error_message}", response_data)
        # Check for N/A for latency values
        self.assertIn("Average Latency: N/A", response_data)
        self.assertIn("Minimum Latency: N/A", response_data)
        self.assertIn("Maximum Latency: N/A", response_data)
        self.assertIn("Standard Deviation: N/A", response_data)
        self.assertIn(IMAGE_URL, response_data) # Check that the image URL is still displayed

if __name__ == '__main__':
    unittest.main()
