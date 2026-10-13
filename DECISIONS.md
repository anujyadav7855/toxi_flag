# ToxiFlag - Decisions

- D1: Virtual environment at .venv inside the project (Python 3.13.15, Windows).
- D2: Day 1 installs only data/ML basics. torch and transformers come later.
- D3: Machine has 7.4 GB RAM, GPU not yet checked. Plan: baseline locally, transformer fine-tuning on a free Colab GPU unless a local CUDA GPU is found.
- D4: data/raw, data/processed and models/ are git-ignored (size and dataset licenses).
- D5: Rejected Kaggle hinglish_cyberbullying_dataset_25000.csv as a research dataset: only 40 unique texts among 25,000 rows (24,960 exact duplicates), so any split leaks. Not used for training or evaluation.
