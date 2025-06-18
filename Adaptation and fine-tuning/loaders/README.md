# 📦 LOADERS — Dataset Utilities

This folder contains scripts for loading and preparing datasets for Whisper-based ASR experiments.
Beware, change the paths to your datasets if you are not able to log-in to the cluster Ponyland. 

## Contents

### `data_loader_cgn2.py`
- Loads the CGN2 (Corpus Gesproken Nederlands) dataset.
- Handles chunking of long audio files into smaller training-ready segments.
- Produces a `datasets.Dataset` object with audio and transcription pairs.

### `data_loader_common.py`
- Loads the language portion of the Common Voice dataset.
- Applies preprocessing for Whisper input (sampling rate, feature extraction).

## Requirements
- `csv`
- `datasets`
- `glob`
- `gzip`
- `librosa`
- `pandas`
- `soundfile`
