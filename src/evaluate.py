import glob
import json
import os

import numpy as np


class Word2VecEvaluator:
    """
    Loads trained Word2Vec embeddings and provides methods for semantic evaluation.
    """

    def __init__(self, model_dir="../saved_models"):
        self.model_dir = model_dir
        self._load_artifacts()
        self._normalize_embeddings()

    def _get_latest_file(self, pattern):
        """
        Finds the most recently created file matching the given pattern.
        """
        search_path = os.path.join(self.model_dir, pattern)
        files = glob.glob(search_path)

        if not files:
            raise FileNotFoundError(f"No files matching '{pattern}' found in {self.model_dir}.")

        # Return the file with the most recent creation/modification time
        return max(files, key=os.path.getctime)

    def _load_artifacts(self):
        """Loads the latest word2idx dictionary and the W1 embedding matrix."""

        vocab_path = self._get_latest_file("word2idx_*.json")
        w1_path = self._get_latest_file("W1_*.npy")

        print(f"Loading vocabulary from: {os.path.basename(vocab_path)}")
        print(f"Loading embeddings from: {os.path.basename(w1_path)}")

        with open(vocab_path, "r") as f:
            self.word2idx = json.load(f)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.embeddings = np.load(w1_path)

    def _normalize_embeddings(self):
        """
        Normalizes all embedding vectors to an L2 norm of 1 to allow for
        efficient cosine similarity computation via dot product.
        """
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.normalized_embeddings = self.embeddings / norms

    def get_similar_words(self, query_word, top_k=10):
        """
        Finds the top K most semantically similar words to the query word.
        """
        if query_word not in self.word2idx:
            return f"Word '{query_word}' not found in vocabulary."

        query_idx = self.word2idx[query_word]
        query_vector = self.normalized_embeddings[query_idx]

        # Compute cosine similarity
        similarities = np.dot(self.normalized_embeddings, query_vector)

        # Retrieve top K indices
        top_indices = np.argsort(similarities)[-(top_k + 1):][::-1]

        results = []
        for idx in top_indices:
            word = self.idx2word[idx]
            if word != query_word:
                results.append((word, similarities[idx]))

        return results[:top_k]

    def get_analogy(self, pos_word1, neg_word, pos_word2, top_k=5):
        """
        Solves analogies of the form: pos_word1 is to neg_word as X is to pos_word2.
        Math: target_vector = pos_word1 - neg_word + pos_word2
        Example: king - man + woman =?
        """
        # Check if all words are in the vocabulary
        for w in [pos_word1, neg_word, pos_word2]:
            if w not in self.word2idx:
                return f"Word '{w}' not found in vocabulary."

        # Fetch the normalized vectors
        vec_p1 = self.normalized_embeddings[self.word2idx[pos_word1]]
        vec_n = self.normalized_embeddings[self.word2idx[neg_word]]
        vec_p2 = self.normalized_embeddings[self.word2idx[pos_word2]]

        # Calculate target vector: king - man + woman
        target_vec = vec_p1 - vec_n + vec_p2

        # Normalize the target vector to safely compute cosine similarity
        target_norm = np.linalg.norm(target_vec)
        if target_norm > 0:
            target_vec = target_vec / target_norm

        # Compute cosine similarity with the entire vocabulary
        similarities = np.dot(self.normalized_embeddings, target_vec)

        # Retrieve indices sorted by the highest similarity
        top_indices = np.argsort(similarities)[::-1]

        results = []
        input_words = {pos_word1, neg_word, pos_word2}

        # Filter out the input words so the model doesn't just return "king" or "woman"
        for idx in top_indices:
            word = self.idx2word[idx]
            if word not in input_words:
                results.append((word, similarities[idx]))
                if len(results) == top_k:
                    break

        return results


if __name__ == "__main__":
    try:
        evaluator = Word2VecEvaluator()
        test_words = ["computer", "king", "water"]

        for test_word in test_words:
            print(f"\nTop 5 words most similar to '{test_word}':")
            similar_words = evaluator.get_similar_words(test_word, top_k=5)

            if isinstance(similar_words, str):
                print(similar_words)
            else:
                for match, score in similar_words:
                    print(f"  {match}: {score:.4f}")

        # Run the Analogy Test
        print("\n" + "=" * 40)
        print("Analogy Test: king - man + woman = ?")
        print("=" * 40)

        analogy_results = evaluator.get_analogy("king", "man", "woman", top_k=5)

        if isinstance(analogy_results, str):
            print(analogy_results)
        else:
            for match, score in analogy_results:
                print(f"  {match}: {score:.4f}")

    except FileNotFoundError as e:
        print(e)
