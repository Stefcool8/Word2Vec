import json
import os
from datetime import datetime

import numpy as np

from src.data_loader import Word2VecDataLoader
from src.word2vec import Word2Vec


FILE_PATH = "data/raw/text8"
MIN_COUNT = 5
WINDOW_SIZE = 5
EMBEDDING_DIM = 100
INITIAL_LEARNING_RATE = 0.001
BATCH_SIZE = 2048
NUM_NEG_SAMPLES = 15
EPOCHS = 15
SAMPLE_THRESHOLD = 1e-3
L2_REG = 1e-4


def main():
    os.makedirs("saved_models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print("--- Step 1: Loading Data ---")
    data_loader = Word2VecDataLoader(
        file_path=FILE_PATH,
        min_count=MIN_COUNT,
        window_size=WINDOW_SIZE,
        sample_threshold=SAMPLE_THRESHOLD
    )
    data_loader.prepare_data()

    print("\n--- Step 2: Initializing Model ---")
    model = Word2Vec(
        vocab_size=data_loader.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        learning_rate=INITIAL_LEARNING_RATE,
        l2_reg=L2_REG
    )

    print("\n--- Step 3: Starting Training ---")

    estimated_total_pairs = len(data_loader.center_words_array) * EPOCHS
    pairs_processed = 0
    min_lr = INITIAL_LEARNING_RATE * 0.0001

    for epoch in range(EPOCHS):
        total_loss = 0.0
        batch_count = 0

        batch_generator = data_loader.generate_batches(BATCH_SIZE)

        for center_words, context_words in batch_generator:
            current_batch_size = len(center_words)

            progress = min(1.0, pairs_processed / estimated_total_pairs)
            current_lr = max(min_lr, INITIAL_LEARNING_RATE * (1.0 - progress))
            model.learning_rate = current_lr

            neg_samples = data_loader.get_negative_samples(current_batch_size, NUM_NEG_SAMPLES)

            loss = model.update(center_words, context_words, neg_samples)
            total_loss += loss
            batch_count += 1
            pairs_processed += current_batch_size

            if batch_count % 500 == 0:
                avg_loss = total_loss / (batch_count * BATCH_SIZE)
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | Batch {batch_count:,} | LR: {current_lr:.5f} | Avg Loss: {avg_loss:.4f}")

        print(f"-> Epoch {epoch + 1} Completed. Total Avg Loss: {total_loss / (batch_count * BATCH_SIZE):.4f}")

        # checkpointing
        epoch_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        print(f"Saving checkpoint for Epoch {epoch + 1}...")
        np.save(f"saved_models/W1_epoch{epoch + 1}_{epoch_timestamp}.npy", model.W1)
        np.save(f"saved_models/W2_epoch{epoch + 1}_{epoch_timestamp}.npy", model.W2)

        # Only need to save the vocabulary once
        if epoch == 0:
            with open(f"saved_models/word2idx_{timestamp}.json", "w") as f:
                json.dump(data_loader.word2idx, f)

    print("\n--- Training Complete! ---")


if __name__ == "__main__":
    main()
