import torch
from torch import nn
import torchvision.models as models # Added for ResNet18
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer
from executorch.exir import to_edge, EdgeCompileConfig
from torch.export import export, ExportedProgram
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import get_symmetric_quantization_config

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def compile_model_to_executorch(
    model: nn.Module,
    example_inputs: tuple,
    output_pte_path: str = None, # Changed default to None
    enable_quantization: bool = False
):
    """
    Optionally performs PTQ, lowers to XNNPACK, and saves the .pte file if output_pte_path is provided.
    Returns the ExecuTorchProgram object.
    """
    # Determine a logging name for the output, even if not saving
    log_name = output_pte_path if output_pte_path else "in-memory program"
    print(f"Starting compilation for {log_name} (Quantization: {enable_quantization})...")

    model.eval()

    current_model_to_export = model

    if enable_quantization:
        print("Step 1: Quantization (Attempting)...")
        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
        quantizer.set_global(quantization_config)

        print("  Exporting model for quantization preparation...")
        exported_program_for_quant_prep = export(model, example_inputs) # Use original model here

        print("  Preparing model for quantization (prepare_pt2e)...")
        try:
            prepared_model = prepare_pt2e(exported_program_for_quant_prep, quantizer)
        except KeyError as e:
            print(f"KeyError during prepare_pt2e: {e}")
            print("This often indicates an issue with quantizer setup or model annotation for the given torch/executorch/torchao versions.")
            try:
                print("Graph of model passed to prepare_pt2e:")
                exported_program_for_quant_prep.graph_module.print_readable()
            except Exception as ge:
                print(f"Could not print graph: {ge}")
            print("Quantization failed, re-raising the error.")
            raise e
        print("  Model prepared for quantization.")

        print("  Calibrating model...")
        for _ in range(3):
            prepared_model(*example_inputs)
        print("  Calibration complete.")

        print("  Converting model to quantized version (convert_pt2e)...")
        quantized_model = convert_pt2e(prepared_model)
        print("  Model converted to quantized version.")
        current_model_to_export = quantized_model
    else:
        print("Step 1: Quantization skipped as per flag.")

    # Proceed with export and lowering using current_model_to_export
    print("Step 2: Exporting and Lowering Model to ExecuTorch for XNNPACK...")

    print(f"  Exporting {'quantized' if enable_quantization else 'original'} model...")
    # Ensure current_model_to_export is an nn.Module or compatible ExportedProgram for the second export
    # If current_model_to_export is an ExportedProgram (from quantizer), it's fine.
    # If it's an nn.Module (original model), it's also fine.
    exported_program: ExportedProgram = export(current_model_to_export, example_inputs)
    print("  Model exported.")

    print("  Lowering to Edge dialect...")
    # _check_ir_validity=False is often necessary for quantized models.
    edge_compile_config = EdgeCompileConfig(_check_ir_validity= (not enable_quantization) )
    edge_manager = to_edge(exported_program, compile_config=edge_compile_config)
    print("  Lowered to Edge dialect.")

    print("  Lowering to XNNPACK delegate...")
    partitioned_program = edge_manager.to_backend(XnnpackPartitioner())
    print("  Lowered to XNNPACK delegate.")

    print("Step 3: Converting to ExecuTorch program...")
    final_program = partitioned_program.to_executorch()
    print("  Converted to ExecuTorch program.")

    if output_pte_path:
        print(f"Step 4: Saving .pte program to {output_pte_path}...")
        try:
            with open(output_pte_path, "wb") as f:
                f.write(final_program.buffer)
            print(f"Successfully saved {output_pte_path}")
        except Exception as e:
            print(f"Error saving .pte file: {e}")
            # Still return the program even if saving failed, or re-raise?
            # For library use, returning is good. For script use, error is clear.
            # Let's re-raise for now if path was given, as user expected a file.
            raise
    else:
        print("Step 4: Skipping file saving as no output_pte_path was provided.")

    return final_program

# compile_resnet18_to_executorch function removed as app.py will call the main compiler directly.

if __name__ == "__main__":
    # import types # Required for MethodType for patching .recompile - This seems not used currently.
    # If it was for prepare_pt2e or other advanced features, it might be needed if those are re-enabled.
    # For now, commenting out if not directly used by the current flow.

    print("Running AOT Compiler Script as __main__ for testing...")

    # --- Compile SimpleModel (kept for testing file output) ---
    print("\n--- Compiling SimpleModel (testing file output) ---")
    simple_model_instance = SimpleModel()
    dummy_inputs_simple = (torch.randn(1, 3, 32, 32),)
    print(f"Created dummy inputs for SimpleModel with shape: {dummy_inputs_simple[0].shape}")

    output_filename_simple_model = "aot_simple_model_xnnpack_main_test.pte"
    print(f"\nAttempting SimpleModel compilation WITHOUT quantization, saving to {output_filename_simple_model}")

    try:
        # Call the modified function, providing a path to test saving
        compiled_program = compile_model_to_executorch(
            model=simple_model_instance,
            example_inputs=dummy_inputs_simple,
            output_pte_path=output_filename_simple_model,
            enable_quantization=False
        )
        if compiled_program:
            print(f"SimpleModel compilation successful. Program returned. Saved to {output_filename_simple_model}")
            # Basic check on the returned program
            print(f"Returned program type: {type(compiled_program)}")
        else:
            # This case should ideally not happen if no exception during compilation
            print("SimpleModel compilation did not return a program object, check for errors.")

    except Exception as e:
        print(f"An error occurred during SimpleModel compilation for __main__ test: {e}")
        import traceback
        traceback.print_exc()

    # Example of calling without saving (could be useful for other tests)
    print("\n--- Compiling SimpleModel (testing in-memory compilation) ---")
    try:
        in_memory_program = compile_model_to_executorch(
            model=simple_model_instance, # Reuse same model and inputs
            example_inputs=dummy_inputs_simple,
            output_pte_path=None, # Explicitly None
            enable_quantization=False
        )
        if in_memory_program:
            print("SimpleModel in-memory compilation successful. Program returned.")
            print(f"Returned program type: {type(in_memory_program)}")
        else:
            print("SimpleModel in-memory compilation did not return a program object.")
    except Exception as e:
        print(f"An error occurred during SimpleModel in-memory compilation test: {e}")
        import traceback
        traceback.print_exc()

    print("\nAOT Compiler Script (__main__) finished.")
