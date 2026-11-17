
import sys
import platform

print("VISIONPORT ENVIRONMENT CHECK")
print("----------------------------")

print("Python Version:", sys.version)
print("Python Executable:", sys.executable)
print("Operating System:", platform.system())

try:
    import torch
    print("PyTorch Version:", torch.__version__)
    print("CUDA Version:", torch.version.cuda)
    print("CUDA Available:", torch.cuda.is_available())
except ImportError:
    print("PyTorch is not installed")

try:
    import ultralytics
    print("Ultralytics Version:", ultralytics.__version__)
except ImportError:
    print("Ultralytics is not installed")

try:
    import fastapi
    print("FastAPI Version:", fastapi.__version__)
except ImportError:
    print("FastAPI is not installed")

print("----------------------------")
print("Environment check completed.")
