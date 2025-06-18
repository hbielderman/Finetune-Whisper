import sys
import os

# Parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np

from huggingface_hub import login

HF_TOKEN = "YOUR_TOKEN_HERE"  # Replace with your Hugging Face token
login(token=HF_TOKEN) 
import os
import csv
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
)

# === CONFIG ===
MAX_DURATION_SEC = 30.0

# === Load JASMIN dataset ===
def load_local_jasmin_dataset(tsv_path, audio_base_path):
    data = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sentence = row["ortographic_transcription"].strip()
            audio_path = os.path.join(audio_base_path, row["audio"])
            if not sentence or not os.path.exists(audio_path):
                continue
            data.append({"audio": audio_path, "sentence": sentence})
    return Dataset.from_list(data).cast_column("audio", Audio(sampling_rate=16000))


# === Filtering ===
def is_valid_sample(batch):
    audio = batch.get("audio", None)
    sentence = batch.get("sentence", "").strip()

    if not audio or not sentence or "array" not in audio or audio["array"] is None:
        return False

    if isinstance(audio["array"], np.ndarray) and audio["array"].size == 0:
        return False

    duration = len(audio["array"]) / audio.get("sampling_rate", 16000)
    return duration <= MAX_DURATION_SEC


# === Preprocessing ===
def prepare_whisper_dataset(batch, processor):
    audio = batch["audio"]
    sentence = batch["sentence"].strip()

    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=16000
    ).input_features[0]

    labels = processor.tokenizer(sentence, truncation=True, max_length=448).input_ids
    labels = [l if l != processor.tokenizer.pad_token_id else -100 for l in labels]
    batch["labels"] = labels
    return batch


# === Collator ===
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch["attention_mask"] = (batch["input_features"] != self.processor.feature_extractor.padding_value).long()

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# === Callbacks ===
class BestModelSaverCallback(TrainerCallback):
    def __init__(self, lang_abbr):
        self.best_score = float("inf")
        self.lang_abbr = lang_abbr

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        score = state.log_history[-1].get("eval_loss", None) if state.log_history else None
        epoch = int(state.epoch)

        if epoch % 10 == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            torch.cuda.empty_cache()
            model.save_pretrained(ckpt_dir)
            print(f"✓ Saving model at epoch {epoch} (checkpoint)")

        if score is not None and score < self.best_score:
            self.best_score = score
            best_dir = os.path.join(args.output_dir, "best")
            torch.cuda.empty_cache()
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"✓ Saving BEST model at epoch {epoch} with eval_loss {score:.4f}")

        state.best_score = self.best_score


# === Main training function ===
def train_whisper(language, lang_abbr, model_dir, tsv_path, clips_path):
    print(f"\n🚀 Starting training for language: {language} ({lang_abbr})")

    processor = WhisperProcessor.from_pretrained(model_dir, language=language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

    print("📂 Loading JASMIN dataset...")
    dataset_raw = load_local_jasmin_dataset(tsv_path, clips_path)

    # Shuffle and split: 80% train, 10% validation, 10% test
    print("🔀 Shuffling and splitting dataset...")
    dataset_raw = dataset_raw.shuffle(seed=42)
    dataset_split = dataset_raw.train_test_split(test_size=0.2, seed=42)  # 80% train, 20% temp
    val_test_split = dataset_split["test"].train_test_split(test_size=0.5, seed=42)  # Split 20% into 10/10
    dataset = DatasetDict({
        "train": dataset_split["train"],
        "validation": val_test_split["train"],
        "test": val_test_split["test"]
    })

    print("\n🔍 Dataset stats:")
    for split in dataset:
        print(f"• {split}: {len(dataset[split])} samples")
        print(dataset[split][0])
    print("\n🧪 Features:")
    print(dataset["train"].features)

    print("🧹 Filtering...")
    for split in dataset:
        dataset[split] = dataset[split].filter(is_valid_sample)

    print("🔄 Preprocessing...")
    for split in ["train", "validation"]:
        dataset[split] = dataset[split].map(
            lambda b: prepare_whisper_dataset(b, processor),
            remove_columns=dataset[split].column_names,
            load_from_cache_file=False
        )

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./whisper_{lang_abbr}",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        learning_rate=4e-6,
        num_train_epochs=70,
        fp16=True,
        logging_dir=f"./whisper_{lang_abbr}/logs",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        tokenizer=processor.tokenizer,
        compute_metrics=None,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=7),
            BestModelSaverCallback(lang_abbr)
        ]
    )

    print("💻 Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("🔥 Training...")
    trainer.train()

    print("💾 Saving final model...")
    trainer.save_model(f"./whisper_{lang_abbr}/final")

    print("📈 Final evaluation on test set...")
    dataset["test"] = dataset["test"].map(
        lambda b: prepare_whisper_dataset(b, processor),
        remove_columns=dataset["test"].column_names,
        load_from_cache_file=False
    )
    metrics = trainer.evaluate(eval_dataset=dataset["test"])
    print(f"\n=== Final Test Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


# === Entry point ===
if __name__ == "__main__":
    # CONFIGURE THESE
    model_dir = "openai/whisper-small"  # or path to your checkpoint
    language = "dutch"
    lang_abbr = "nl"
    base_path = "/vol/bigdata3/datasets3/dutch_child_audio"
    tsv_path = f"{base_path}/jasmin/tsv/ref_comp-p-dlg_nl_age7-11_nat.tsv"
    clips_path = base_path  # audio path is relative to this

    train_whisper(language, lang_abbr, model_dir, tsv_path, clips_path)
