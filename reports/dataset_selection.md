# Dataset Selection (Day 2)

## Selected dataset
- File: data/raw/hinglish_tweets.csv, made by src/data/extract_hinglish.py from combined_hate_speech_dataset.csv (rows with source == hate_speech_tsv).
- Rows: 4,574. Columns: text, label.
- Labels: 0 = Normal Speech 2,914 (63.7%); 1 = Hate Speech 1,660 (36.3%).
- Script: Roman-script Hinglish tweets. Devanagari 0.0%, missing 0, exact duplicates 0, duplicates after lowercase/trim 0, same text with different labels 0.
- Length: mean 20.0 words, median 19, min 1, max 67.

## Provenance and license (not fully verified)
- Class counts match Bohra et al. (2018) as reported in a later paper (4,575 tweets; 1,661 hate; 2,914 normal) within one row. Provenance was not verified independently.
- The original GitHub repo (deepanshu1995/HateSpeech-Hindi-English-Code-Mixed-Social-Media-Text) releases only tweet IDs and labels (4,114 lines in id_annotated.tsv); its README says tweet text is available from the authors on request. Repo license: GPL-3.0.
- TODO: license of the Kaggle compilation (PRISM, sharduldhekane); permission or confirmation from the original authors.

## Label definition
- Original classes: Hate Speech vs Normal Speech. Used as binary. The positive class is called "hate speech", not "toxic"; the toxic/non-toxic mapping needs justification in the report.

## Known limitations
- Label noise seen in a read of about 40 random samples: some tweets using the word "hate" about non-group targets are labelled Hate, and some clearly abusive tweets are labelled Normal. Not a full audit.
- Small size (4,574); a 15% test split is about 690 rows, so metric variance will be high.
- Single platform (Twitter). Some tweets look like Roman Urdu mixed with Hindi and English.
- Near-duplicate check (beyond exact and lowercase/trim) not yet done.

## Rejected candidates
| Candidate | Reason (from actual inspection) |
|---|---|
| hinglish_cyberbullying_dataset_25000.csv | 25,000 rows but only 40 unique texts (24,960 duplicates); any split leaks |
| Hinglish_CodeMixed_Data.csv (9,354 rows, labels -1/0) | Label 0 was voice-assistant commands in a 30-sample check, label -1 was profanity chat; label confounded with domain; 1,152 duplicates after lowercase/trim |
| Bohra GitHub repo alone | Tweet IDs only, no text |
| PRISM english_dataset (15,000) | English, not Hinglish |
| PRISM hindi_dataset (9,767) | 100% Devanagari |
| PRISM profanity_list (209) | Single words, all label 1 |
| unified_dataset_raw.csv, final_cleaned_dataset.csv (4,860 rows) | Labels rewritten by an LLM; 1,965 (40.4%) changed, Hinglish 369 of 742 (49.7%); not ground truth |
| hard_examples.csv | Malformed rows (parser error at line 518); first rows are English |
| Yugrathee28/Hinglish-dataset (Hugging Face) | Not downloaded; dataset card describes a 5,000-row teaser with weak labels and commercial-use restrictions |
| textdetox/multilingual_toxicity_dataset (hin split) | Not evaluated |
