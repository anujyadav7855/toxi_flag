"""src/check_env.py - verifies Python, packages, RAM and (optional) GPU."""
import importlib
import platform
import sys

print(f"Python  : {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")

REQUIRED = ["pandas", "numpy", "sklearn", "matplotlib", "joblib", "psutil"]
OPTIONAL = ["torch", "transformers", "streamlit"]

failed = []
print("\nRequired packages:")
for name in REQUIRED:
    try:
        mod = importlib.import_module(name)
        print(f"  OK       {name:<14} {getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"  MISSING  {name}")
        failed.append(name)

print("\nOptional packages (not needed on Day 1):")
for name in OPTIONAL:
    try:
        mod = importlib.import_module(name)
        print(f"  found    {name:<14} {getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"  not installed  {name}")

try:
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"\nRAM     : {ram_gb:.1f} GB")
except ImportError:
    pass

try:
    import torch
    print(f"GPU     : {'CUDA available - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA GPU detected'}")
except ImportError:
    print("GPU     : torch not installed yet, so GPU not checked")

print("\nRESULT  :", "FAILED - fix missing packages" if failed else "PASSED")
sys.exit(1 if failed else 0)