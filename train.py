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
import numpy as np
import torch
import csv
import torchaudio
import librosa
import torch
import evaluate

# Set Whisper model
whisper_model = "openai/whisper-medium"

# Set output dir
output_dir_name = "/scratch/hbielderman/whisper-finetuned"

# Set modifications
speed_perturbation = True
if speed_perturbation:
    perturb = SpeedPerturbation(orig_freq=16000, factors=[0.9, 1.0, 1.1])
spec_augmentation = True

# Path to your Common Voice folder
cv_base = "/vol/bigdata3/corpora3/common-voice-dutch/cv-corpus-11.0-2022-09-21/nl"
tsv_file = f"{cv_base}/train.tsv" 
clips_folder = f"{cv_base}/clips"

# Get whisper model
model = WhisperForConditionalGeneration.from_pretrained(whisper_model, cache_dir="/tmp/huggingface")
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model, cache_dir="/tmp/huggingface")
tokenizer = WhisperTokenizer.from_pretrained(whisper_model, cache_dir="/tmp/huggingface", language="Dutch", task="transcribe")
processor = WhisperProcessor.from_pretrained(whisper_model, cache_dir="/tmp/huggingface", language="Dutch", task="transcribe")

# Get (audio path, transcription) format
data = []
with open(tsv_file, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        audio_path = f"{clips_folder}/{row['path']}"
        data.append({
            "audio": audio_path,
            "sentence": row["sentence"]
        })

common_voice_train = Dataset.from_list(data)
dataset = common_voice_train.shuffle(seed=42)
train_subset = dataset.select(range(len(dataset) // 2))
common_voice_train = train_subset.cast_column("audio", Audio(sampling_rate=16000))

# Repeat for test set
test_tsv = f"{cv_base}/test.tsv"
test_data = []
with open(test_tsv, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        audio_path = f"{clips_folder}/{row['path']}"
        test_data.append({
            "audio": audio_path,
            "sentence": row["sentence"]
        })

common_voice_test = Dataset.from_list(test_data)
common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16000))

# Combine data
common_voice = DatasetDict({
    "train": common_voice_train,
    "test": common_voice_test,
})


def prepare_trainset(batch):
    audio = batch["audio"]
    waveform = audio["array"]

    if speed_perturbation:
      # Apply speed perturbation
      waveform = torch.tensor(audio["array"], dtype=torch.float32)
      perturbed_waveform, _ = perturb(waveform)
      waveform = perturbed_waveform.squeeze(0).numpy()

    # compute log-Mel input features
    features = feature_extractor(waveform, sampling_rate=audio["sampling_rate"]).input_features[0]

    if spec_augmentation:
      # Apply spectral augmentation
      features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
      augmented_tensor = spec_augment_pytorch.spec_augment(features_tensor)
      features = augmented_tensor.squeeze(0).numpy()
    
    batch["input_features"] = features

    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def prepare_testset(batch):
    audio = batch["audio"]

    # compute log-Mel input features
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice["train"] = common_voice["train"].map(prepare_trainset, remove_columns=common_voice.column_names["train"], num_proc=2)
common_voice["test"] = common_voice["test"].map(prepare_testset, remove_columns=common_voice.column_names["test"], num_proc=2)

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

# Prepare evaluation metrics
metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# Set training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir_name,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    learning_rate=3e-5,
    warmup_steps=100,
    num_train_epochs=2,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# Set trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
processor.save_pretrained(training_args.output_dir)

# Train and save model
trainer.train()
trainer.save_model(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
