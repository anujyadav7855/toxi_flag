"""setup_structure.py - one-time script to create the ToxiFlag project skeleton.

Run from inside the toxiflag/ folder:  python setup_structure.py
Existing files are never overwritten.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FOLDERS = [
    "data/raw", "data/processed", "notebooks",
    "src/data", "src/preprocessing", "src/models",
    "src/disagreement", "src/evaluation", "src/inference",
    "app", "tests", "models", "reports/figures", "reports/metrics",
]

FILES = {
    "requirements.txt": """pandas
numpy
scikit-learn
matplotlib
jupyter
joblib
psutil
# Added in later phases (not installed on Day 1):
# torch
# transformers
# streamlit
""",
    ".gitignore": """venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
.vscode/
# Data and model artifacts are not committed (size / license)
data/raw/
data/processed/
models/
"""}
