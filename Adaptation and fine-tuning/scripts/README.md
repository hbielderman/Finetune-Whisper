# 🛠️ SCRIPTS — Preprocessing, Analysis & Evaluation

This folder includes utility scripts for analyzing datasets, preparing training input, and evaluating model predictions.
Beware, change the paths to your datasets if you are not able to log-in to the cluster Ponyland. 

## Contents

### Data Inspection & Analysis
- **`analyze_metadata.py`**  
  Analyzes dataset metadata (e.g., speaker gender, age) and prints distributions.

- **`investigating.py`**  
  Utility for inspecting dataset samples or debugging data/model behaviour.

### Preprocessing & Collation
- **`chunking.py`**  
  Chunks long audio recordings (e.g., from CGN2) into smaller pieces suitable for Whisper.

- **`combining_docs.py`**  
  Merges transcription files or multiple datasets into a single unified format.

- **`data_collator.py`**  
  Defines a custom data collator for Whisper fine-tuning. Handles label padding and BOS token trimming.

- **`finetuning_preprocessing.py`**  
  Preprocesses data (tokenization, audio feature extraction) for Whisper model training.

### Evaluation
- **`jasmin_test_child.py`**  
  Evaluates a fine-tuned Whisper model on Dutch child speech from JASMIN.

- **`jasmin_test_whisper.py`**  
  Evaluates a base Whisper model (`openai/whisper-small`, etc.) on JASMIN.

## Notes
- Supports Dutch text normalization (e.g., mapping numerals).
- Evaluation computes WER and CER using `jiwer`.

## Requirements
- see requirments from loaders/
- `jiwer`
- `json`
- `matplotlib`
- `numpy`
- `re`
- `transformers`
- `torch`
- `typing`
- `whisperx`
