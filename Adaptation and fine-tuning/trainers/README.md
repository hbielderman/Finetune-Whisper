# 🏋️ TRAINERS — Whisper Fine-Tuning Scripts

This folder contains training scripts using Hugging Face's `Seq2SeqTrainer` for Whisper-based ASR.
Beware, change the paths to your datasets if you are not able to log-in to the cluster Ponyland. 

## Contents

### `whisper_trainer_pro.py`
- Trains Whisper on the Common Voice dataset.
- Uses `forced_decoder_ids` for consistent decoding prompts.
- Logs metrics: loss, WER, CER.

### `whisper_trainer_jasmin.py`
- Fine-tunes Whisper on the JASMIN Dutch child speech dataset.
- Uses a custom `DataCollator` to handle label padding and BOS trimming.
- Designed for speaker-specific or domain adaptation tasks.

### `whisper_trainer_CGN2.py`
- Fine-tunes Whisper on the CGN2 dataset with pre-chunked audio segments.
- Similar structure to the JASMIN trainer, adapted to CGN audio metadata.

## Features
- `Seq2SeqTrainingArguments` setup for generation-based evaluation.
- Support for GPU training, generation config (`num_beams`, `no_repeat_ngram_size`, etc.).
- Evaluation uses decoded predictions and `jiwer` metrics.

## Requirements
- see requirments from scripts/
- `huggingface_hub`
