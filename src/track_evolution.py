import glob
import json
import os
import re

import numpy as np


def track_analogy_evolution(pos_word1="king", neg_word="man", pos_word2="woman", top_k=3, model_dir="../saved_models"):
    print(f"Loading vocabulary from {model_dir}...")

    # Find the most recent vocabulary file
    vocab_files = glob.glob(os.path.join(model_dir, "word2idx_*.json"))
    if not vocab_files:
        print(f"No vocabulary file found in {model_dir}.")
        return

    vocab_path = max(vocab_files, key=os.path.getctime)
    print(f" -> Using vocabulary file: {os.path.basename(vocab_path)}")

    with open(vocab_path, "r") as f:
        word2idx = json.load(f)

    idx2word = {idx: word for word, idx in word2idx.items()}

    # Check if words are in the vocabulary
    for w in [pos_word1, neg_word, pos_word2]:
        if w not in word2idx:
            print(f"Word '{w}' not found in vocabulary.")
            return

    idx_p1 = word2idx[pos_word1]
    idx_n = word2idx[neg_word]
    idx_p2 = word2idx[pos_word2]
    input_words = {pos_word1, neg_word, pos_word2}

    print("\nLocating epoch checkpoints...")
    # Find all W1 checkpoint files
    checkpoint_files = glob.glob(os.path.join(model_dir, "W1_epoch*.npy"))
    if not checkpoint_files:
        print("No epoch checkpoints found.")
        return

    # Sort files numerically by epoch number using regex
    def get_epoch_num(filename):
        match = re.search(r"epoch(\d+)", filename)
        return int(match.group(1)) if match else 0

    checkpoint_files.sort(key=get_epoch_num)

    print("\n" + "=" * 65)
    print(f"Tracking Evolution: {pos_word1} - {neg_word} + {pos_word2} = ?")
    print("=" * 65 + "\n")

    for cp in checkpoint_files:
        epoch = get_epoch_num(cp)
        filename = os.path.basename(cp)
        embeddings = np.load(cp)

        # Normalize the embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized = embeddings / norms

        # Fetch vectors and apply the algebraic formula
        vec_p1 = normalized[idx_p1]
        vec_n = normalized[idx_n]
        vec_p2 = normalized[idx_p2]

        target_vec = vec_p1 - vec_n + vec_p2

        # Normalize the resulting target vector
        target_norm = np.linalg.norm(target_vec)
        if target_norm > 0:
            target_vec = target_vec / target_norm

        # Compute similarities across the entire vocabulary
        similarities = np.dot(normalized, target_vec)
        top_indices = np.argsort(similarities)[::-1]

        print(f"--- Epoch {epoch:<2} | File: {filename} ---")

        results_found = 0
        for idx in top_indices:
            word = idx2word[idx]
            if word not in input_words:
                # Formatted for clean vertical alignment
                print(f"  {word:<15}: {similarities[idx]:.4f}")
                results_found += 1
                # Show top K results per epoch to keep the output readable
                if results_found == top_k:
                    break
        print()


if __name__ == "__main__":
    track_analogy_evolution("king", "man", "woman")

    # How the model learned geography or grammar
    track_analogy_evolution("paris", "france", "germany")
    track_analogy_evolution("swimming", "swim", "walk")
