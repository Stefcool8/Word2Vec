import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import json
from datetime import datetime

import numpy as np

from src2.data_loader import Word2VecDataLoader
from src2.word2vec import Word2Vec

FILE_PATH = "../data/raw/text8"
MIN_COUNT = 5
WINDOW_SIZE = 5
EMBEDDING_DIM = 100
INITIAL_LEARNING_RATE = 0.1   # typical word2vec starting LR
BATCH_SIZE = 256
NUM_NEG_SAMPLES = 10             # Mikolov often used 5
EPOCHS = 10
SAMPLE_THRESHOLD = 1e-3
L2_REG = 0.0 #1e-6

def main():
    os.makedirs("saved_models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print("--- Step 1: Loading Data ---")
    data_loader = Word2VecDataLoader(
        file_path=FILE_PATH,
        min_count=MIN_COUNT,
        window_size=WINDOW_SIZE,
        sample_threshold=SAMPLE_THRESHOLD,
        use_unigram_table=False,
        unigram_table_size=int(1e7)
    )
    data_loader.prepare_data()

    print("\n--- Step 2: Initializing Model ---")
    model = Word2Vec(
        vocab_size=data_loader.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        learning_rate=INITIAL_LEARNING_RATE,
        l2_reg=L2_REG,
        clip_val=None
    )

    print("\n--- Step 3: Starting Training ---")
    estimated_total_pairs = len(data_loader.center_words_array) * EPOCHS
    pairs_processed_global = 0

    for epoch in range(EPOCHS):
        total_loss_sum = 0.0     # sum of (batch_avg_loss * batch_size) over epoch
        pairs_processed_epoch = 0
        batch_count = 0

        batch_generator = data_loader.generate_batches(BATCH_SIZE)

        for center_words, context_words in batch_generator:
            current_batch_size = len(center_words)
            if current_batch_size == 0:
                continue

            # simple linear decay schedule (clipped)
            progress = min(1.0, pairs_processed_global / max(1, estimated_total_pairs))
            current_lr = max(1e-6, INITIAL_LEARNING_RATE * (1.0 - progress))
            model.learning_rate = current_lr

            # get negatives sized (batch, num_neg)
            neg_samples = data_loader.get_negative_samples(current_batch_size, NUM_NEG_SAMPLES)

            # model.update returns average loss per positive pair (per-example average)
            batch_avg_loss = model.update(center_words, context_words, neg_samples)

            # accumulate weighted by batch size so global average is correct
            total_loss_sum += float(batch_avg_loss) * current_batch_size
            pairs_processed_epoch += current_batch_size
            pairs_processed_global += current_batch_size
            batch_count += 1

            # logging periodically
            if batch_count % 500 == 0:
                avg_loss_so_far = total_loss_sum / max(1, pairs_processed_epoch)
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | Batch {batch_count:,} | "
                    f"LR: {current_lr:.5f} | Avg Loss (per pair): {avg_loss_so_far:.4f}"
                )

        # end of epoch
        epoch_avg_loss = total_loss_sum / max(1, pairs_processed_epoch)
        print(f"-> Epoch {epoch + 1} Completed. Epoch Avg Loss (per pair): {epoch_avg_loss:.4f}")

        # Checkpointing
        epoch_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        print(f"Saving checkpoint for Epoch {epoch + 1}...")
        np.save(f"saved_models/W1_epoch{epoch + 1}_{epoch_timestamp}.npy", model.W1)
        np.save(f"saved_models/W2_epoch{epoch + 1}_{epoch_timestamp}.npy", model.W2)

        # Save vocab once (first epoch)
        if epoch == 0:
            with open(f"saved_models/word2idx_{timestamp}.json", "w") as f:
                json.dump(data_loader.word2idx, f)

    print("\n--- Training Complete! ---")


if __name__ == "__main__":
    main()
