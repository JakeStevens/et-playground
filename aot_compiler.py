import torch
from torch import nn
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
    output_pte_path: str = "model_xnnpack.pte",
    enable_quantization: bool = False # New flag
):
    """
    Optionally performs PTQ, lowers to XNNPACK, and saves the .pte file.
    """
    print(f"Starting compilation for {output_pte_path} (Quantization: {enable_quantization})...")

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

    print(f"Step 4: Saving .pte program to {output_pte_path}...")
    try:
        with open(output_pte_path, "wb") as f:
            f.write(final_program.buffer)
        print(f"Successfully saved {output_pte_path}")
    except Exception as e:
        print(f"Error saving .pte file: {e}")
        raise

    return final_program

if __name__ == "__main__":
    import types # Required for MethodType for patching .recompile

    print("Running AOT Compiler Script...")
    simple_model = SimpleModel()
    dummy_inputs = (torch.randn(1, 3, 32, 32),)
    print(f"Created dummy inputs with shape: {dummy_inputs[0].shape}")

    # Example: Compile without quantization (default)
    output_filename_no_quant = "aot_model_xnnpack_no_quant_main.pte" # Different name for main run
    print(f"\nAttempting compilation WITHOUT quantization to {output_filename_no_quant}")
    try:
        compile_model_to_executorch(
            simple_model,
            dummy_inputs,
            output_filename_no_quant,
            enable_quantization=False
        )
        print(f"Compilation without quantization successful.")
    except Exception as e:
        print(f"An error occurred during non-quantized compilation: {e}")

    print("\nAOT Compiler Script finished.")
