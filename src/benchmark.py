import glob
import json
import os
import time

import numpy as np


class Word2VecBenchmark:
    def __init__(self, model_dir="../saved_models", eval_file="../data/eval/questions-words.txt"):
        self.model_dir = model_dir
        self.eval_file = eval_file
        self._load_artifacts()
        self._normalize_embeddings()

    def _get_latest_file(self, pattern):
        search_path = os.path.join(self.model_dir, pattern)
        files = glob.glob(search_path)
        if not files:
            raise FileNotFoundError(f"No files matching '{pattern}' found.")
        return max(files, key=os.path.getctime)

    def _load_artifacts(self):
        vocab_path = self._get_latest_file("word2idx_*.json")
        w1_path = self._get_latest_file("W1_*.npy")

        print(f"Loading vocabulary from: {os.path.basename(vocab_path)}")
        print(f"Loading embeddings from: {os.path.basename(w1_path)}")

        with open(vocab_path, "r") as f:
            self.word2idx = json.load(f)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.embeddings = np.load(w1_path)

    def _normalize_embeddings(self):
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.normalized_embeddings = self.embeddings / norms

    def run_benchmark(self):
        if not os.path.exists(self.eval_file):
            print(f"Error: Could not find benchmark file at {self.eval_file}")
            return

        print(f"\nStarting benchmark on {self.eval_file}...")
        start_time = time.time()

        with open(self.eval_file, "r") as f:
            lines = f.readlines()

        current_category = None
        results = {}

        # Track overall stats
        total_seen = 0
        total_correct = 0
        total_skipped = 0

        for line in lines:
            line = line.strip().lower()
            if not line:
                continue

            # New category header
            if line.startswith(":"):
                current_category = line[1:].strip()
                results[current_category] = {"correct": 0, "total": 0}
                continue

            words = line.split()
            if len(words) != 4:
                continue

            w1, w2, w3, target = words

            # If any word is not in the vocabulary, the question is skipped
            if any(w not in self.word2idx for w in [w1, w2, w3, target]):
                total_skipped += 1
                continue

            # Fetch normalized vectors
            vec_w1 = self.normalized_embeddings[self.word2idx[w1]]
            vec_w2 = self.normalized_embeddings[self.word2idx[w2]]
            vec_w3 = self.normalized_embeddings[self.word2idx[w3]]

            # w2 - w1 + w3 = target
            # (e.g., king - boy + girl = queen)
            target_vec = vec_w2 - vec_w1 + vec_w3

            # Compute similarities with the entire vocabulary
            similarities = np.dot(self.normalized_embeddings, target_vec)

            input_indices = {self.word2idx[w] for w in [w1, w2, w3]}

            # Find the index of the highest similarity that isn't an input word
            best_idx = None
            for idx in np.argsort(similarities)[::-1]:
                if idx not in input_indices:
                    best_idx = idx
                    break

            # Tally the results
            is_correct = (self.idx2word[best_idx] == target)

            if is_correct:
                results[current_category]["correct"] += 1
                total_correct += 1

            results[current_category]["total"] += 1
            total_seen += 1

        # Report
        print("\n" + "=" * 50)
        print("GOOGLE ANALOGY BENCHMARK RESULTS")
        print("=" * 50)

        for category, stats in results.items():
            if stats["total"] > 0:
                acc = (stats["correct"] / stats["total"]) * 100
                print(f"{category:>30}: {acc:5.1f}%  ({stats['correct']}/{stats['total']})")

        print("-" * 50)
        overall_acc = (total_correct / total_seen) * 100 if total_seen > 0 else 0
        print(f"{'OVERALL ACCURACY':>30}: {overall_acc:5.1f}%  ({total_correct}/{total_seen})")
        print(f"{'SKIPPED (Out of Vocab)':>30}: {total_skipped} questions")
        print(f"Completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    benchmark = Word2VecBenchmark()
    benchmark.run_benchmark()
