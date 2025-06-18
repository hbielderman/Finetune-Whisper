import os
from datasets import Dataset, DatasetDict, Audio

def load_myst_children_dataset(base_dir="/vol/bigdata3/corpora3/MyST/ldc_downloader/myst_child_conv_speech/data"):
    splits = ["train", "development", "test"]
    data = {split: [] for split in splits}

    for split in splits:
        split_path = os.path.join(base_dir, split)
        print(f"🔍 Scanning {split} data in {split_path} ...")

        # Walk recursively
        for root, _, files in os.walk(split_path):
            # Find all audio files (.flac or .wav)
            audio_files = [f for f in files if f.endswith(".flac")]
            for audio_file in audio_files:
                audio_path = os.path.join(root, audio_file)
                # Match transcript by replacing extension with .trn
                transcript_file = audio_file.rsplit('.', 1)[0] + ".trn"
                transcript_path = os.path.join(root, transcript_file)

                if not os.path.exists(transcript_path):
                    print(f"⚠️ Missing transcript for {audio_path}")
                    continue

                # Read transcript text
                with open(transcript_path, 'r', encoding='utf-8') as ftrn:
                    transcript = ftrn.read().strip()

                # Parse student_id and session_id from path parts
                # e.g. /data/train/012030/myst_012030_2013-12-03_10-12-56_MX_2.1/...
                parts = root.split(os.sep)
                student_id = parts[-2] if len(parts) >= 2 else "unknown"
                session_id = parts[-1] if len(parts) >= 1 else "unknown"

                data[split].append({
                    "audio": audio_path,
                    "sentence": transcript,
                    "student_id": student_id,
                    "partition": split,
                    "gender": "unknown",  
                    "age": None           
                })

        print(f"✅ Loaded {len(data[split])} samples from {split}")

    # Convert to HuggingFace DatasetDict
    dataset_dict = DatasetDict({
        split: Dataset.from_list(data[split]) for split in splits
    })

    # Cast audio column for streaming + resampling support
    for split in splits:
        dataset_dict[split] = dataset_dict[split].cast_column("audio", Audio(sampling_rate=16000))

    return dataset_dict

