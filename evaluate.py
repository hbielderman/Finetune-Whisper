from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Dataset, Audio
from torchaudio.transforms import SpeedPerturbation
from specAugment import spec_augment_pytorch
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import WhisperForConditionalGeneration
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from jiwer import wer
from datasets import DatasetDict
import torch
import csv
import torchaudio
import numpy as np
import librosa
import torch
import evaluate

# Set model dir
model_dir_name = "/whisper-finetuned"

# Path to test data folder
cv_base = "/vol/bigdata3/datasets3/dutch_child_audio"
tsv_file = f"{cv_base}/jasmin/tsv/ref_comp-p-dlg_nl_age7-11_nat.tsv"
clips_folder = f"{cv_base}"

# Get (audio path, transcription) format
data = []
with open(tsv_file, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        audio_path = f"{clips_folder}/{row['audio']}"
        data.append({
            "audio": audio_path,
            "sentence": row["ortographic_transcription"]
        })

jasmin_test = Dataset.from_list(data)
jasmin_test = jasmin_test.cast_column("audio", Audio(sampling_rate=16000))
jasmin = DatasetDict({
    "test": jasmin_test,
})

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir_name)
tokenizer = WhisperTokenizer.from_pretrained(model_dir_name, language="Dutch", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_dir_name)
processor = WhisperProcessor.from_pretrained(model_dir_name)

def prepare_testset(batch):
    audio = batch["audio"]

    audio_input = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    batch["input_features"] = audio_input["input_features"][0]

    text_input = processor.tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    batch["labels"] = text_input.input_ids[0] 

    return batch

jasmin['test'] = jasmin['test'].map(prepare_testset, remove_columns=jasmin.column_names['test'])

# Prepare evaluation metrics
def compute_wer(predictions, references):
    return wer(references, predictions)
def compute_metrics(p):
    predictions = processor.batch_decode(p.predictions, skip_special_tokens=True)
    references = processor.batch_decode(p.label_ids, skip_special_tokens=True)
    return {"wer": compute_wer(predictions, references)}

# Set training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./output",
    per_device_eval_batch_size=4,
    predict_with_generate=True, 
    logging_dir="./logs"
)

# Set trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics
)

# Evaluate on the training dataset
train_results = trainer.evaluate(eval_dataset=jasmin['test'])

# Print out the WER result
print(f"WER on the training set: {train_results['eval_wer']}")
