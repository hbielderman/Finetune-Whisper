from transformers import WhisperTokenizer
import matplotlib.pyplot as plt
import numpy as np

def inspect_transcript_token_lengths(dataset, processor):
    lengths = []
    for i, example in enumerate(dataset):
        sentence = example["sentence"]
        tokens = processor.tokenizer(
            sentence,
            truncation=False,  # important: we want real lengths
            return_tensors=None,
        ).input_ids
        lengths.append(len(tokens))
    
    # Plot
    plt.hist(lengths, bins=50, color='skyblue')
    plt.title("Distribution of Tokenized Transcript Lengths")
    plt.xlabel("Number of tokens")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    print(f"📊 Min: {min(lengths)}")
    print(f"📊 Max: {max(lengths)}")
    print(f"📊 Mean: {sum(lengths) / len(lengths):.2f}")
    print(f"📊 95th percentile: {np.percentile(lengths, 95):.2f}")
    print(f"📊 99th percentile: {np.percentile(lengths, 99):.2f}")
