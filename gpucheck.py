import torch
import sys

def check_gpu_capabilities():
    print("Python Version:", sys.version)
    print("PyTorch Version:", torch.__version__)
    
    # Check CUDA availability
    print("\nCUDA Available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        # GPU Details
        print("\nGPU Device Details:")
        print("Device Name:", torch.cuda.get_device_name(0))
        print("Total GPU Memory:", torch.cuda.get_device_properties(0).total_memory / (1024**3), "GB")
        
        # Memory Allocation Test
        try:
            # Attempt to allocate increasing memory
            test_sizes = [1, 2, 4, 8, 16]  # GB
            for size in test_sizes:
                try:
                    x = torch.cuda.FloatTensor(int(size * 256e6))
                    del x
                    print(f"Successfully allocated {size} GB")
                except RuntimeError as e:
                    print(f"Could not allocate {size} GB: {e}")
        except Exception as e:
            print("Memory allocation test failed:", e)
    else:
        print("No CUDA-capable GPU detected.")

# Run diagnostic
check_gpu_capabilities()