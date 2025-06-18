import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import csv
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from datasets import Audio, Dataset, DatasetDict
from jiwer import cer, wer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import string
import re

#empty cache
torch.cuda.empty_cache()

# === Paths & loading ===
#model_dir = "./whisper_nl_CGN/best"
model_dir = "./whisper_nl/best"
#model_dir = "./whisper_nl_child/best"

cv_base = "/vol/bigdata3/datasets3/dutch_child_audio"
#tsv_path = f"{cv_base}/jasmin/tsv/ref_comp-p-dlg_nl_age7-11_nat.tsv"
tsv_path = f"{cv_base}/jasmin/tsv/ref_comp-q-read_nl_age7-11_nat.tsv"
clips_path = f"{cv_base}"

# === Load raw data ===
print("📂 Loading dataset...")
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

# === Load processor and model ===
print("🔄 Loading model and processor...")
processor = WhisperProcessor.from_pretrained(model_dir, language="dutch", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_dir)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="dutch", task="transcribe") #NEW
print(f"🔍 Loaded model: {model.config._name_or_path} with {model.num_parameters()} parameters")

# === Preprocessing function ===
def preprocess(batch):
    audio = batch["audio"]

    # Extract features (returns a dict with 'input_features' key with shape [1, seq_len, feat_dim])
    audio_features = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    )
    batch["input_features"] = audio_features["input_features"][0]  # remove batch dim

    # Tokenize labels, pad/truncate to max_length=60
    text_tokens = processor.tokenizer(
        batch["sentence"],
        #padding="max_length",
        padding="longest",
        truncation=True,
        #max_length=60,
        return_tensors="pt",
    )
    batch["labels"] = text_tokens.input_ids[0]  # remove batch dim
    #print(f"🔎 Token length: {len(text_tokens.input_ids[0])}")
    return batch

# Check if using GPU
if torch.cuda.is_available():
    print("🚀 Using GPU for processing!")
else:
    print("💻 Using CPU for processing, this may be slow!")


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

def normalize_dutch_text(text):
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    for word, digit in DUTCH_NUMERAL_MAP.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    return text

# Apply preprocessing
print("🔄 Preprocessing dataset...")
#dataset["test"] = dataset["test"].map(preprocess, remove_columns=dataset["test"].column_names)

# === Data collator for batching ===
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Extract input features for padding
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch["attention_mask"] = (batch["input_features"] != self.processor.feature_extractor.padding_value).long()  # NEW
        #print("🔢 Input shape:", batch["input_features"].shape)
        # Input shape: torch.Size([4, 80, 3000])

        # Extract labels and pad them properly
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding tokens by -100 to ignore loss on them
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS token if present (Whisper specific)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        print("🎯 Label shape:", labels.shape)
        batch["labels"] = labels

        return batch

collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# === Generation configuration ===
forced_decoder_ids = processor.get_decoder_prompt_ids(language="dutch", task="transcribe")
model.config.forced_decoder_ids = forced_decoder_ids
model.generation_config.language = "<|nl|>"
model.generation_config.task = "transcribe"

# === Manual Evaluation ===
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Using GPU for evaluation!")
else:
    device = torch.device("cpu")
    print("💻 Using CPU for evaluation, this may be slow!")
    
model.to(device)
model.eval()

print("🔍 Starting manual evaluation...")

all_pred_texts = []
all_label_texts = []

for i, sample in enumerate(dataset["test"]):
    audio = sample["audio"]
    sentence = sample["sentence"]

    # Prepare input tensor and send to device
    inputs = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features.to(device)
    attention_mask = (inputs != processor.feature_extractor.padding_value).long().to(device)
    
    with torch.no_grad():
        # Generate with forced_decoder_ids
        predicted_ids = model.generate(
            inputs,
            max_length=70,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_decoder_ids,
            repetition_penalty=1.2,  
            no_repeat_ngram_size=3,
            num_beams=5,
            early_stopping=True
        )

    pred_text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True, normalize=False)[0]

    all_pred_texts.append(pred_text)
    all_label_texts.append(sentence)

    # Print first 3 predictions and references
    if i < 5:
        print(f"📝 Sample prediction #{i+1}: {pred_text}")
        print(f"📝 Reference #{i+1}: {sentence}")

# Calculate metrics with jiwer
# Normalize predictions and references before scoring
normalized_preds = [normalize_dutch_text(p) for p in all_pred_texts]
normalized_refs = [normalize_dutch_text(r) for r in all_label_texts]

# Calculate metrics
wer_score = wer(normalized_refs, normalized_preds)
cer_score = cer(normalized_refs, normalized_preds)

print(f"\n📊 Final Evaluation Results:\nWER: {wer_score:.4f} | CER: {cer_score:.4f}")

# Save results to JSON
results = {
    "wer": wer_score,
    "cer": cer_score,
    "num_samples": len(all_pred_texts),
}

with open("./eval_out/eval_results_manual.json", "w") as f:
    json.dump(results, f, indent=2)
