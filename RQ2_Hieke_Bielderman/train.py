import numpy as np
import torch
import csv
import torchaudio
import librosa
import torch
import evaluate
import re
import random
import torchaudio.sox_effects as sox_effects
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Dataset, Audio
from torchaudio.transforms import SpeedPerturbation
from specAugment import spec_augment_pytorch
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import WhisperForConditionalGeneration
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict

# Settings
whisper_model = "openai/whisper-small"
output_dir_name = "./model"
cv_base = "/vol/bigdata3/corpora3/CGN_NL_Chunks/comp_o/"
tsv_file = f"{cv_base}/text/comp_o.csv"
clips_folder = f"{cv_base}/audio"
speed_perturbation = True
spec_augmentation = True

# Get Whisper model
model = WhisperForConditionalGeneration.from_pretrained(whisper_model, cache_dir="/tmp/huggingface")
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model, cache_dir="/tmp/huggingface")
tokenizer = WhisperTokenizer.from_pretrained(whisper_model, cache_dir="/tmp/huggingface", language="Dutch", task="transcribe")
processor = WhisperProcessor.from_pretrained(whisper_model, cache_dir="/tmp/huggingface", language="Dutch", task="transcribe")

# Get data
MAX_FILES = 10_000
all_data = []
with open(tsv_file, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=",")
    for row in reader:
        all_data.append(row)
data = []
for row in all_data:
    sentence = row["Transcript"]
    audio_path = f"{row['AudioFilePath']}"
    data.append({"audio": audio_path, "sentence": sentence})
dataset = Dataset.from_list(data).shuffle(seed=42)
dataset = dataset.select(range(min(MAX_FILES, len(dataset))))
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
CGN_train = split_dataset["train"].cast_column("audio", Audio(sampling_rate=16000))
CGN_test = split_dataset["test"].cast_column("audio", Audio(sampling_rate=16000))

# Combine data
CGN = DatasetDict({
    "train": CGN_train,
    "test": CGN_test,
})

speed_factors = [0.9, 1.0, 1.1]

def apply_speed_perturbation(waveform: torch.Tensor, speed: float) -> torch.Tensor:
    effects = [
        ["speed", f"{speed}"],
        ["rate", "16000"]
    ]
    augmented_waveform, _ = sox_effects.apply_effects_tensor(waveform.unsqueeze(0), 16000, effects)
    return augmented_waveform.squeeze(0)


def prepare_trainset(batch):
    augmented_features = []
    augmented_labels = []

    for i in range(len(batch["audio"])):
        waveform = torch.tensor(batch["audio"][i]["array"], dtype=torch.float32)
        sentence = batch["sentence"][i]

        for speed in speed_factors:
            # Apply speed perturbation (define this function)
            if speed != 1.0 and speed_perturbation:
                augmented_waveform = apply_speed_perturbation(waveform, speed)
            else:
                augmented_waveform = waveform

            audio_array = augmented_waveform.numpy()

    # compute log-Mel input features
    features = feature_extractor(waveform, sampling_rate=audio["sampling_rate"]).input_features[0]

    if spec_augmentation:
      # Apply spectral augmentation
      features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).transpose(1, 2).cpu()
      augmented_tensor = spec_augment_pytorch.spec_augment(features_tensor, time_warping_para=5, frequency_masking_para=20, time_masking_para=30)
      features = augmented_tensor.transpose(1, 2).squeeze(0).numpy()

    batch["input_features"] = features
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def prepare_testset(batch):
    audio_arrays = [a["array"] for a in batch["audio"]]
    sampling_rate = batch["audio"][0]["sampling_rate"]
    batch["input_features"] = feature_extractor(audio_arrays, sampling_rate=sampling_rate).input_features
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

CGN["train"] = CGN["train"].map(prepare_trainset, batched=True, batch_size=8, num_proc=4, remove_columns=CGN["train"].column_names)
CGN["test"] = CGN["test"].map(prepare_testset, remove_columns=CGN.column_names["test"], num_proc=4,  batched=True, batch_size=16)

# Define datacollator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Define evaluation metrics
metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Set training argsuments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir_name,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=4,
    learning_rate=4e-6,
    warmup_steps=100,
    num_train_epochs=20,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,
    generation_max_length=256,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False
)

# Set trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=CGN["train"],
    eval_dataset=CGN["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
processor.save_pretrained(training_args.output_dir)

# Train and save model
trainer.train()
trainer.save_model(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
