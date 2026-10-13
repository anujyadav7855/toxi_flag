# ToxiFlag - Decisions

- D1: Virtual environment at .venv inside the project (Python 3.13.15, Windows).
- D2: Day 1 installs only data/ML basics. torch and transformers come later.
- D3: Machine has 7.4 GB RAM, GPU not yet checked. Plan: baseline locally, transformer fine-tuning on a free Colab GPU unless a local CUDA GPU is found.
- D4: data/raw, data/processed and models/ are git-ignored (size and dataset licenses).
- D5: Rejected Kaggle hinglish_cyberbullying_dataset_25000.csv as a research dataset: only 40 unique texts among 25,000 rows (24,960 exact duplicates), so any split leaks. Not used for training or evaluation.
- D6: Rejected Hinglish_CodeMixed_Data.csv (9,354 rows, labels -1/0) as a primary dataset: in a 30-sample check per class, every label-0 text was a voice-assistant command (timer/alarm/traffic) and label -1 texts were profanity-containing chat, so the label is confounded with source/domain. Several label -1 texts use profanity as filler, so -1 does not mean toxic. 1,152 duplicates after lower/trim. Sorted by label. Source page, license and label definition not verified.
- D7: Bohra et al. dataset (GitHub repo deepanshu1995/HateSpeech-Hindi-English-Code-Mixed-Social-Media-Text) contains only tweet IDs and labels (4114 lines in id_annotated.tsv; paper reports 4575 tweets); tweet text must be requested from the authors per README. Repo license is GPL-3.0 (data reuse terms to be confirmed). Not usable until text is obtained. Tweet re-hydration rejected as unreliable.
