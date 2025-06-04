import unittest
import torch
import os
# Use the specific loading mechanism identified earlier
from executorch.extension.pybindings import _portable_lib as portable_lib
# Need ExecuTorchModule for isinstance check, assuming it's still relevant from previous successful test
from executorch.runtime import ExecuTorchModule


# Assuming aot_compiler.py is in the same directory or PYTHONPATH is set up
from aot_compiler import compile_model_to_executorch, SimpleModel

class TestAotCompiler(unittest.TestCase):

    def setUp(self):
        self.model = SimpleModel()
        # Match input size used in aot_compiler's main block for consistency
        self.example_inputs = (torch.randn(1, 3, 32, 32),)
        self.pte_path = "test_model_xnnpack_no_quant.pte" # Explicitly name for non-quantized test

        # Clean up any existing .pte file before running a test
        if os.path.exists(self.pte_path):
            os.remove(self.pte_path)

    def tearDown(self):
        # Clean up the .pte file after tests
        if os.path.exists(self.pte_path):
            os.remove(self.pte_path)

    def test_compile_and_run_model_non_quantized(self):
        # 1. Compile the model using the function from aot_compiler
        #    Explicitly set enable_quantization=False, or rely on default.
        print(f"Test: Compiling model (non-quantized) to {self.pte_path}...")
        compile_model_to_executorch(
            self.model,
            self.example_inputs,
            self.pte_path,
            enable_quantization=False # Explicitly testing the non-quantized path
        )
        print("Test: Compilation finished.")

        # 2. Verify the .pte file is created
        self.assertTrue(os.path.exists(self.pte_path), f".pte file not found at {self.pte_path}")
        print(f"Test: Verified .pte file exists at {self.pte_path}")

        # 3. Load the generated .pte file
        print("Test: Loading .pte file...")

        try:
            with open(self.pte_path, "rb") as f:
                pte_data = f.read()
            # Using _load_for_executorch_from_buffer as identified in previous successful test run
            executor_module = portable_lib._load_for_executorch_from_buffer(pte_data)

            # It's good practice to check if the loaded module is of the expected type,
            # though portable_lib._load_for_executorch_from_buffer directly returns the C++ bound ExecuTorchModule.
            # The ExecuTorchModule for type checking is imported from executorch.runtime.
            if not isinstance(executor_module, ExecuTorchModule):
                 print(f"Warning: Loaded object type is {type(executor_module)}, not strictly an instance of the imported ExecuTorchModule from executorch.runtime. This might be okay if it's a compatible C++ bound type.")
                 # Depending on strictness, could fail here or proceed. For now, proceed.

        except Exception as e:
            self.fail(f"Failed to load .pte file: {e}")
        print("Test: .pte file loaded successfully.")

        # 4. Create a new set of random inputs
        new_inputs_list = [torch.randn(1, 3, 32, 32)]
        print(f"Test: Created new random inputs with shape: {new_inputs_list[0].shape}")

        # 5. Execute the loaded model
        print("Test: Executing loaded model...")
        try:
            # Call forward method as identified previously
            # The loaded module from _portable_lib is directly callable for its 'forward' method.
            if callable(executor_module.forward):
                outputs = executor_module.forward(inputs=new_inputs_list)
            # Fallback for other ways if direct call is not the case for some modules
            elif hasattr(executor_module, 'run_method'):
                 outputs = executor_module.run_method("forward", tuple(new_inputs_list))
            else:
                raise AttributeError("Loaded module has no callable 'forward' method or 'run_method'.")

        except Exception as e:
            self.fail(f"Model execution failed: {e}")

        print(f"Test: Model execution finished. Output: {outputs}")

        # 6. Assert that the output is not None and has the expected shape/type
        self.assertIsNotNone(outputs, "Model output is None.")
        # The _load_for_executorch_from_buffer path returns a list of tensors
        self.assertTrue(isinstance(outputs, list), "Model output is not a list.")
        self.assertTrue(len(outputs) > 0, "Model output list is empty.")
        output_tensor = outputs[0]
        self.assertTrue(isinstance(output_tensor, torch.Tensor), "Output is not a torch.Tensor.")

        self.assertEqual(output_tensor.shape, (1, 10), f"Output shape mismatch: expected (1, 10), got {output_tensor.shape}")
        print(f"Test: Verified output tensor type and shape {output_tensor.shape}.")

if __name__ == '__main__':
    unittest.main()
