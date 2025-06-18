import os
import glob
import gzip
import pandas as pd
import librosa
import soundfile as sf
from datasets import Dataset, DatasetDict, Audio

CGN_ROOT = "/vol/bigdata/corpora2/CGN2/data"
ANNOT_DIR = os.path.join(CGN_ROOT, "annot/text/ort")
AUDIO_DIR = os.path.join(CGN_ROOT, "audio/wav")
CHUNKED_AUDIO_DIR = "/vol/tensusers6/rchissich/ASR/cleaned_data/audio/chunked_first30"
TARGET_COMPONENTS = ["comp-o", "comp-a"]

def safe_read_ort(ort_file):
    try:
        with gzip.open(ort_file, "rt", encoding="iso-8859-1") as f:
            lines = f.readlines()
            if not lines:
                return None
            sentence = lines[-1].strip()
            return sentence
    except Exception as e:
        print(f"Skipping file {ort_file} due to error: {e}")
        return None

def extract_first_chunk(audio_path, output_dir, chunk_duration=30, target_sr=16000):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(audio_path)
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path):
        return output_path

    audio, sr = librosa.load(audio_path, sr=target_sr)
    end_sample = int(chunk_duration * sr)
    end_sample = min(end_sample, len(audio))
    chunk = audio[:end_sample]
    sf.write(output_path, chunk, target_sr)
    
    return output_path

def load_cgn2_dataset(lang_abbr="nl", language="dutch"):
    print(f"🔍 Loading CGN2 dataset for language: {lang_abbr} ({language})")
    data = []

    for component in TARGET_COMPONENTS:
        component_dir = os.path.join(ANNOT_DIR, component, lang_abbr)
        ort_files = glob.glob(os.path.join(component_dir, "*.ort.gz"))

        print(f"Component {component} -> Found {len(ort_files)} ort files")

        for ort_file in ort_files:
            basename = os.path.basename(ort_file).replace(".ort.gz", "")
            original_audio_path = os.path.join(AUDIO_DIR, component, lang_abbr, basename + ".wav")

            if not os.path.exists(original_audio_path):
                print(f"Missing audio for: {basename}")
                continue

            sentence = safe_read_ort(ort_file)
            if sentence is None:
                continue

            chunked_component_dir = os.path.join(CHUNKED_AUDIO_DIR, component, lang_abbr)
            chunked_audio_path = extract_first_chunk(original_audio_path, chunked_component_dir)

            data.append({
                "audio": chunked_audio_path,
                "sentence": sentence
            })

    print(f"✅ Loaded {len(data)} CGN2 samples.")

    if not data:
        raise ValueError("No data loaded from CGN2!")

    dataset = Dataset.from_pandas(pd.DataFrame(data))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

    dataset_dict = DatasetDict({
        "train": dataset["train"],
        "test": dataset["test"]
    })

    return dataset_dict
