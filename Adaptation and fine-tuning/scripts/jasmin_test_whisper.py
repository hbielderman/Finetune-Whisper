import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import csv
from datasets import Dataset, Audio, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from jiwer import wer, cer
import matplotlib.pyplot as plt
import json
import re
import string

# === Paths & loading ===
model_dir = "openai/whisper-small" # base model
#model_dir = "./whisper_nl_ft/best"
#model_dir = "./whisper_nl_no/best"

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
forced_decoder_ids = processor.get_decoder_prompt_ids(language="dutch", task="transcribe")
print(f"🔍 Loaded model: {model.config._name_or_path} with {model.num_parameters()} parameters"
)
#print(forced_decoder_ids)
#tokenizer = WhisperTokenizer.from_pretrained(model_dir)
#print(tokenizer.convert_ids_to_tokens([50271, 50359, 50363]))

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

#check if using GPU
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
dataset["test"] = dataset["test"].map(preprocess, remove_columns=dataset["test"].column_names)

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
        batch["attention_mask"] = (batch["input_features"] != self.processor.feature_extractor.padding_value).long() #NEW
        #print("🔢 Input shape:", batch["input_features"].shape)
        #Input shape: torch.Size([4, 80, 3000])

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
model.config.forced_decoder_ids = forced_decoder_ids
model.generation_config.language = "<|nl|>"
model.generation_config.task = "transcribe"

# === Metrics ===
def compute_metrics(eval_pred):
    pred_ids = eval_pred.predictions
    label_ids = eval_pred.label_ids

    # Mask out -100 so padding doesn't affect decoding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Normalize both predictions and references
    norm_preds = [normalize_dutch_text(t) for t in pred_texts]
    norm_labels = [normalize_dutch_text(t) for t in label_texts]

    # Compute WER and CER
    wer_score = wer(norm_labels, norm_preds)
    cer_score = cer(norm_labels, norm_preds)

    print(f"📝 Sample decoded predictions: {pred_texts[:3]}")
    print(f"📝 Sample references: {label_texts[:3]}")

    return {
        "wer": wer_score,
        "cer": cer_score,
    }

# === TrainingArguments & Trainer for evaluation ===
args = Seq2SeqTrainingArguments(
    output_dir="./eval_out",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    logging_dir="./logs",
)

class WhisperTrainerWithForcedDecoderIds(Seq2SeqTrainer):
    def prediction_step(
        self, model, inputs, prediction_loss_only, ignore_keys=None
    ):
        # Explicitly inject forced_decoder_ids into generation
        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
            #generation_kwargs=generation_kwargs, #NOT for ft
        )

#trainer = Seq2SeqTrainer(
trainer = WhisperTrainerWithForcedDecoderIds(
    model=model,
    args=args,
    tokenizer=processor.tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# === Run evaluation ===
print("🔍 Starting evaluation...")
results = trainer.evaluate(eval_dataset=dataset["test"])
# Save the results to a JSON file so we can plot later
with open("./eval_out/eval_results_new.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n📊 Evaluation Results:\nWER: {results['eval_wer']:.4f} | CER: {results['eval_cer']:.4f} | Loss: {results['eval_loss']:.4f}")
