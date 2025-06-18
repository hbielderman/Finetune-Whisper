#https://colab.research.google.com/github/Vaibhavs10/fast-whisper-finetuning/blob/main/Whisper_w_PEFT.ipynb/#scrollTo=8326221e-ec13-4731-bb4e-51e5fc1486c5
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        #print("🔢 Input shape:", batch["input_features"].shape)


        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
    
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        #print("🎯 Label shape:", labels.shape)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        max_len = labels.shape[1]
        if max_len > 448:
            print(f"⚠️ Warning: unusually long label sequence of length {max_len}")

        #print("📦 Batch size:", len(features), batch)
        return batch
    
    