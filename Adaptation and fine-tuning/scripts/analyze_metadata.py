from collections import Counter
import numpy as np
import re

def analyze_metadata(dataset_split, split_name="train"):
    """
    Prints age and gender statistics for a dataset split.
    
    Args:
        dataset_split: A HuggingFace Dataset (e.g., dataset["train"])
        split_name (str): Name of the split (for labeling output)
    """
    print(f"\n 🔍 Analyzing metadata for {split_name.upper()}...")
    # AGE STATISTICS
    ages = [example.get("age") for example in dataset_split if example.get("age") not in [None, "", "NA"]]
    if ages:
        age_counts = Counter(ages)
        print(f"[{split_name.upper()}] Age distribution:")
        for age_group, count in sorted(age_counts.items()):
            print(f"  - {age_group}: {count}")
    else:
        print(f"[{split_name.upper()}] No valid age data found.")

    # GENDER DISTRIBUTION
    genders = [
        sample["gender"].lower()
        for sample in dataset_split
        if "gender" in sample and sample["gender"] and sample["gender"].lower() not in {"", "unknown"}
    ]

    if genders:
        gender_counts = Counter(genders)
        print(f"📊 [{split_name.upper()}] Gender distribution:")
        for gender, count in gender_counts.items():
            print(f"   - {gender}: {count}")
    else:
        print(f"⚠️ [{split_name.upper()}] No valid gender data found.")
