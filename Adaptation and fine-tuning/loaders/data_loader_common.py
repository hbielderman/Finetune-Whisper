import os
import csv
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, Audio, concatenate_datasets


def clean_test_tsv_skip_bad_rows(filepath, clean_dir):
    import os
    import pandas as pd

    os.makedirs(clean_dir, exist_ok=True)
    df = pd.read_csv(filepath, sep="\t")

    # Replace empty or invalid 'age' with 0
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0).astype(int)

    # Filter out rows with 'Benchmark' or other invalid strings in ANY column
    def is_row_valid(row):
        for val in row:
            if isinstance(val, str) and val.lower() == 'benchmark':
                return False
        return True

    valid_df = df[df.apply(is_row_valid, axis=1)]

    # Save cleaned TSV
    filename = os.path.basename(filepath).replace(".tsv", "_cleaned.tsv")
    clean_path = os.path.join(clean_dir, filename)
    valid_df.to_csv(clean_path, sep="\t", index=False)

    print(f"🧹 Cleaned test.tsv saved with {len(valid_df)} valid rows out of {len(df)} total rows")

    return clean_path

# Balance each split by gender (train, validation, test)
def balance_by_gender(dataset_split, max_per_gender=15000):
     # Keep only male and female entries
    filtered = dataset_split.filter(lambda x: x.get("gender") in ["male", "female"]).shuffle(seed=123)

    # Separate and shuffle by gender
    males = filtered.filter(lambda x: x["gender"] == "male").shuffle(seed=1)
    females = filtered.filter(lambda x: x["gender"] == "female").shuffle(seed=2)

    # Select equal samples per gender (up to max)
    N = min(len(males), len(females), max_per_gender)
    balanced = Dataset.from_list(males.select(range(N)).to_list() + females.select(range(N)).to_list())
    print(f"✅ Balanced to {N} male + {N} female samples")

    return balanced.shuffle(seed=42)


def balance_female_plus_others(dataset_split, max_per_gender=5000):
    females = dataset_split.filter(lambda x: x.get("gender") == "female").shuffle(seed=42)
    non_females = dataset_split.filter(lambda x: x.get("gender") != "female").shuffle(seed=42)
    
    n_females = len(females)
    needed_non_females = max_per_gender - n_females
    if needed_non_females > len(non_females):
        print(f"⚠️ Not enough non-female samples ({len(non_females)}) to reach target length ({max_per_gender}).")
        needed_non_females = len(non_females)
    
    non_female_subset = non_females.select(range(needed_non_females))
    
    combined = Dataset.from_list(females.to_list() + non_female_subset.to_list()).shuffle(seed=42)
    
    print(f"✅ Dataset: {n_females} females + {needed_non_females} non-females = {len(combined)} total samples")
    return combined

def load_common_voice_dataset(lang_abbr: str, language_name: str, dataset_version="13_0"):
    dataset_name = f"mozilla-foundation/common_voice_{dataset_version}"
    common_voice = DatasetDict()

    # Train and test splits
    common_voice["train"] = load_dataset(dataset_name, lang_abbr, split="train+validation")
    common_voice["test"] = load_dataset(dataset_name, lang_abbr, split="test")
    
    # ⚖️ Balance both train and test by gender
    print("🔄 Balancing training and test sets by gender...")
    common_voice["train"] = balance_by_gender(common_voice["train"], max_per_gender=15000)
    common_voice["test"] = balance_female_plus_others(common_voice["test"], max_per_gender=5000)

    # Remove unnecessary columns
    columns_to_remove = [
        "accent", "client_id", "down_votes", "locale", "path",
        "segment", "up_votes", "variant"
    ]
    for split in common_voice:
        common_voice[split] = common_voice[split].remove_columns(columns_to_remove) #NEW!
    
    # Correct sampling rate for the models (from 48 to 16kHz)
    for split in common_voice:
        common_voice[split] = common_voice[split].cast_column("audio", Audio(sampling_rate=16_000)) #NEW!

    return common_voice

def load_local_common_voice_dataset(lang_abbr: str, language_name: str):
    if lang_abbr == "nl":
        dataset_path = "/vol/bigdata3/corpora3/common-voice-dutch/cv-corpus-11.0-2022-09-21/nl"
    elif lang_abbr == "en":
        dataset_path = "/vol/bigdata3/corpora3/common-voice-english/cv-corpus-10.0-2022-07-04/en"
    elif lang_abbr == "es":
        dataset_path = "/vol/bigdata3/corpora3/common-voice-spanish/cv-corpus-8.0-2022-01-19/es"
    else:
        raise ValueError(f"Unsupported language abbreviation: {lang_abbr}")

    # Load from local CSV files
    data_files = {
        "train": os.path.join(dataset_path, "train.tsv"),
        "validation": os.path.join(dataset_path, "dev.tsv"),
        "test": os.path.join(dataset_path, "test.tsv")
    }

    # 🔍 Debug problematic rows in test.tsv (before load_dataset)
    test_tsv_path = data_files["test"]
    print(f"🔍 Inspecting test TSV: {test_tsv_path}")
    with open(test_tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        print(f"📑 Columns: {reader.fieldnames}")
        for i, row in enumerate(reader):
            print(f"🔹 Row {i + 1}: {row}")
            if i == 1:
                break
    
    # Only clean the test split
    clean_dir = f"./cleaned_data/{lang_abbr}"
    cleaned_test_path = clean_test_tsv_skip_bad_rows(data_files["test"], clean_dir)

    # Use original train and validation, but cleaned test
    cleaned_data_files = {
        "train": data_files["train"],
        "validation": data_files["validation"],
        "test": cleaned_test_path
    }

    dataset = load_dataset("csv", data_files=cleaned_data_files, delimiter="\t")

    print("🔄 Balancing training set by gender...")
    dataset["train"] = balance_by_gender(dataset["train"], max_per_gender=15000)
    dataset["validation"] = balance_female_plus_others(dataset["validation"], max_per_gender=5000)
    dataset["test"] = balance_female_plus_others(dataset["test"], max_per_gender=5000)

    # Remove unwanted columns
    columns_to_remove = [
        "accents", "client_id", "down_votes", "locale",
        "segment", "up_votes", "variant"
    ]
    for split in dataset:
        cols_to_remove = [col for col in columns_to_remove if col in dataset[split].column_names]
        dataset[split] = dataset[split].remove_columns(cols_to_remove)

    def add_audio_path(example):
        example["audio"] = os.path.join(dataset_path, "clips", example["path"])
        return example

    dataset = dataset.map(add_audio_path)
    # Load audio from 'clips' folder
    print(f"📂 Resampling at the right rate")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    return dataset
