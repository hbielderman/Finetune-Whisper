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

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
)
from loaders.data_loader_common import load_common_voice_dataset, load_local_common_voice_dataset
from scripts.metrics import compute_metrics
from scripts.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from scripts.analyze_metadata import analyze_metadata


class BestModelSaverCallback(TrainerCallback):
    def __init__(self, lang_abbr):
        self.best_score = float("inf")
        self.lang_abbr = lang_abbr

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        metrics = state.log_history[-1] if state.log_history else {}

        score = metrics.get("eval_loss", None)
        epoch = int(state.epoch)

        if epoch % 10 == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            torch.cuda.empty_cache()
            model.save_pretrained(ckpt_dir)
            print(f"✓ Saving model for language nl at epoch {epoch} (checkpoint)")

        if score is not None and score < self.best_score:
            self.best_score = score
            best_dir = os.path.join(args.output_dir, "best")
            torch.cuda.empty_cache()
            model.save_pretrained(f"./whisper_nl/best")
            print(f"✓ Saving best model for language nl at epoch {epoch} with eval_loss {score:.4f}")

        # Store it on trainer's state for external access
        state.best_score = self.best_score

def compute_duration_stats(dataset):
    durations = []
    for sample in dataset:
        audio = sample.get("audio", None)
        if audio is None or audio.get("array") is None:
            continue
        length = len(audio["array"])
        sr = audio.get("sampling_rate", 16000)  # default to 16k if missing
        durations.append(length / sr)
    if durations:
        print(f"📊 Average: {np.mean(durations):.2f} sec")
        print(f"📊 Median:  {np.median(durations):.2f} sec")
        print(f"📊 Max:     {np.max(durations):.2f} sec")
        print(f"📊 Min:     {np.min(durations):.2f} sec")
    else:
        print("⚠️ No valid audio samples to compute duration stats.")

MAX_DURATION_SEC = 20.0

def is_valid_sample(batch):
    audio = batch.get("audio", None)
    sentence = batch.get("sentence", "").strip()

    if (
        audio is None or
        "array" not in audio or
        audio["array"] is None or
        len(audio["array"]) == 0 or
        not sentence
    ):
        return False
    #return True
    duration = len(audio["array"]) / audio.get("sampling_rate", 16000) # NEW
    return duration <= MAX_DURATION_SEC

def prepare_whisper_dataset(batch, processor):
    #assert isinstance(batch["labels"], list), "Labels should be a list of token IDs"
    #assert isinstance(batch["labels"][0], int), "Each label should be an int"

    audio = batch.get("audio", None)
    sentence = batch.get("sentence", "").strip()

    if (
        audio is None or
        "array" not in audio or
        audio["array"] is None or
        len(audio["array"]) == 0 or
        not sentence
    ):
        raise ValueError("Invalid or empty audio or sentence")

    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=16000
    ).input_features[0]
    
    labels = processor.tokenizer(
        sentence, truncation=True, max_length=200
    ).input_ids

    # Replace padding token ID with -100 (so they are ignored in the loss)
    labels = [l if l != processor.tokenizer.pad_token_id else -100 for l in labels]
    batch["labels"] = labels
    '''
    batch["labels"] = processor.tokenizer(
        sentence, truncation=True, max_length=200
    ).input_ids'''

    return batch


def train_whisper(language, lang_abbr):
    print(f"\n🚀 Starting training for language: {language} ({lang_abbr})")
    model_name = "openai/whisper-small"
    task = "transcribe"

    # Load processor
    print("📦 Loading feature extractor and tokenizer...")
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)

    # Load and prepare dataset
    print("📚 Loading dataset...")
    dataset = load_local_common_voice_dataset(lang_abbr, language)
    print(f"📥 Loaded Common Voice dataset: {dataset} ({lang_abbr})")
    print(f"📊 Train samples: {len(dataset['train'])}, Test samples: {len(dataset['test'])}")

    original_train = dataset["train"].select(range(len(dataset["train"])))
    print(dataset["test"][0])
    analyze_metadata(original_train, "train")
 
    # Check training split
    compute_duration_stats(dataset["train"])    
    filtered_train = dataset["train"].filter(is_valid_sample)
    compute_duration_stats(filtered_train) # NEW 
    processed_train = filtered_train.map(
        lambda b: prepare_whisper_dataset(b, processor),
        remove_columns=filtered_train.column_names,
        #num_proc=1,
        load_from_cache_file=False
        #keep_in_memory=True  
    )
    
    original_test = dataset["test"].select(range(len(dataset["test"])))
    analyze_metadata(original_test, "test")
    # Check testing split
    compute_duration_stats(dataset["test"])    
    filtered_test = dataset["test"].filter(is_valid_sample) 
    compute_duration_stats(filtered_test) # NEW
    processed_test = filtered_test.map(
        lambda b: prepare_whisper_dataset(b, processor),
        remove_columns=filtered_test.column_names,
        #num_proc=1,
        load_from_cache_file=False
        #keep_in_memory=True     
    )

    print("🧠 Loading model...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.return_dict = True
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
        greater_is_better=False,           # WER: lower is better
        load_best_model_at_end=True,
        learning_rate=4e-6,
        num_train_epochs=70,
        fp16=True,
        logging_dir=f"./whisper_{lang_abbr}/logs"
    )

    print("🏋️ Initializing trainer...")
    best_callback = BestModelSaverCallback(lang_abbr) #NEW
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_test,
        data_collator=collator,
        #processing_class=processor,
        tokenizer=processor.tokenizer,
        #tokenizer=tokenizer,
        compute_metrics=None, #lambda p: compute_metrics(p, processor.tokenizer),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=7),
            #BestModelSaverCallback(lang_abbr)
            best_callback
        ]
    )
    
    print("🧹 Clearing CUDA cache...")
    torch.cuda.empty_cache()

    # Print the device being used for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Training on device: {device}")

    print("🔥 Starting training loop...")
    trainer.train()
    print(f"🏆 Best eval_loss during training: {best_callback.best_score:.4f}")

    # Save final model
    print("💾 Saving final model...")
    trainer.save_model(f"./whisper_{lang_abbr}/final")

    torch.cuda.empty_cache()
    
    # Final evaluation
    print("📈 Running final evaluation...")
    metrics = trainer.evaluate()
    print(f"\n==== Final Metrics for {language.upper()} ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return metrics

if __name__ == "__main__":
    train_whisper(language="dutch", lang_abbr="nl") # change the language and lang_abbr as needed