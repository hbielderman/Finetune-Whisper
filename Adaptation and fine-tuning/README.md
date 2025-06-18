# Automatic Speech Recognition (ASR) Training with Whisper

This repository supports training, fine-tuning, and evaluating OpenAI’s Whisper models on Dutch speech datasets, especially child-directed speech (e.g., JASMIN) and adult speech (e.g., CGN2, Common Voice).

---
## 📂 Folder Structure

### `LOADERS/` — Dataset & Model Loaders

This folder contains Python scripts for loading audio datasets used across experiments.

### `SCRIPTS/` — Utility & Evaluation Scripts

General-purpose scripts for data analysis, preprocessing, and evaluation.

### `TRAINERS/` — Whisper Training Scripts

Scripts using Hugging Face’s `Trainer` (or subclasses) for supervised training.

---

## 🔐 Hugging Face Configuration & Dataset Access


Before running training scripts, make sure you are authenticated with Hugging Face and have access to any gated datasets.

Create or use your existing Hugging Face account and log in via CLI:

```bash
huggingface-cli login
```


Make sure you have accepted access to this gated dataset on Hugging Face:

[https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)

After that, you can launch the training script.


Then launch the training script.

---

## 💬 Supported Datasets

- **JASMIN** — Dutch child speech (ages 7–11)
- **CGN2** — Corpus Gesproken Nederlands (adult Dutch)
- **Common Voice** — Mozilla’s multilingual open-source corpus

---

## 🧪 Evaluation Features

- **Metrics**: WER and CER calculated via `jiwer`
- **Text normalization**: Dutch numeral conversion, punctuation cleanup
- **Compatibility**: Works with both base and fine-tuned Whisper models

---

## 🚀 Typical Workflow

1. **Train a model**:  
   Use one of the `whisper_trainer_*.py` scripts

2. **Evaluate performance**:  
   With `jasmin_test_*.py` or Hugging Face `Trainer.evaluate()`

---

## 📎 Requirements

- Python 3.8+
- see requirments of the folders.

---

## 📫 Contact

For questions or collaboration inquiries, feel free to open an issue or get in touch [@RobCTs](https://github.com/RobCTs/ASR/).
