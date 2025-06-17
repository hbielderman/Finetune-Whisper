import os
import re
import torch
import csv
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Audio, Dataset, DatasetDict
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from jiwer import cer, wer
from jiwer import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, wer, cer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.empty_cache()

# Paths
model_dir = "/vol/tensusers8/hbielderman/final-pre-cgn-combined"

cv_base = "/vol/bigdata3/datasets3/dutch_child_audio"
tsv_path = f"{cv_base}/jasmin/tsv/ref_comp-q-read_nl_age7-11_nat.tsv"
clips_path = f"{cv_base}"

# Loading raw data
print("Loading dataset...")
data = []

with open(tsv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        data.append({
            "audio": f"{clips_path}/{row['audio']}",
            "sentence": row["ortographic_transcription"],
        })

dataset_test = Dataset.from_list(data).cast_column("audio", Audio(sampling_rate=16000))
dataset = DatasetDict({"test": dataset_test})

# Data preprocessing functions
basic_transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip()
])

DUTCH_NUMERAL_MAP = {
    "nul": "0",
    "één": "1",
    "twee": "2",
    "drie": "3",
    "vier": "4",
    "vijf": "5",
    "zes": "6",
    "zeven": "7",
    "acht": "8",
    "negen": "9",
    "tien": "10",
    "elf": "11",
    "twaalf": "12",
    "dertien": "13",
    "veertien": "14",
    "vijftien": "15",
    "zestien": "16",
    "zeventien": "17",
    "achttien": "18",
    "negentien": "19",
    "twintig": "20"
}

def normalize_dutch_numerals(text: str) -> str:
    for word, digit in DUTCH_NUMERAL_MAP.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    return text

def full_normalize(text):
    # Handle both string and list of strings
    if isinstance(text, list):
        return [full_normalize(t) for t in text]
    if not isinstance(text, str):
        raise TypeError(f"Expected string or list of strings, got {type(text)}")

    text = basic_transform(text)
    text = normalize_dutch_numerals(text)
    return text

# Load model ===
print("Loading model and processor...")
processor = WhisperProcessor.from_pretrained(model_dir)
model = WhisperForConditionalGeneration.from_pretrained(model_dir)
print(f"Loaded model: {model.config._name_or_path} with {model.num_parameters()} parameters")

# Data collator for batching
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Extract input features for padding
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch["attention_mask"] = (batch["input_features"] != self.processor.feature_extractor.padding_value).long()

        # Extract labels and pad them properly
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding tokens by -100 to ignore loss on them
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS token if present (Whisper specific)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Generation configuration
model.generation_config.language = "<|nl|>"
model.generation_config.task = "transcribe"

# Evaluation
print("Starting evaluation...")
model.to(device)
model.eval()
all_pred_texts = []
all_label_texts = []

for i, sample in enumerate(dataset["test"]):
    audio = sample["audio"]
    sentence = sample["sentence"]
    duration = len(audio["array"]) / audio["sampling_rate"]

    # Prepare input tensor and send to device
    inputs = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features.to(device)
    attention_mask = (inputs != processor.feature_extractor.padding_value).long().to(device)

    with torch.no_grad():
        # Generate with forced_decoder_ids
        predicted_ids = model.generate(
            inputs,
            max_length=256,
            attention_mask=attention_mask,
            repetition_penalty=1.2,    
            no_repeat_ngram_size=2,
        )

    pred_text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True, normalize=False)[0]
    all_pred_texts.append(pred_text)
    all_label_texts.append(sentence)
  
    # Print first 30 predictions and references
    if i < 30:
        print(f"Sample prediction #{i+1}: {pred_text}")
        print(f"Reference #{i+1}: {sentence}")

normalized_preds = [full_normalize(p) for p in all_pred_texts]
normalized_refs = [full_normalize(r) for r in all_label_texts]
wer_score = wer(normalized_refs, normalized_preds)
print(f"\nFinal evaluation results:\nWER: {wer_score:.4f}")

