import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class Word2VecVisualizer:
    def __init__(self, model_dir="../saved_models"):
        self.model_dir = model_dir
        self._load_artifacts()
        self._normalize_embeddings()

    def _get_latest_file(self, pattern):
        search_path = os.path.join(self.model_dir, pattern)
        files = glob.glob(search_path)
        if not files:
            raise FileNotFoundError(f"No files matching '{pattern}' found in {self.model_dir}.")
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

    def get_similar_words(self, query_word, top_k=15):
        if query_word not in self.word2idx:
            return []

        query_idx = self.word2idx[query_word]
        query_vector = self.normalized_embeddings[query_idx]
        similarities = np.dot(self.normalized_embeddings, query_vector)
        top_indices = np.argsort(similarities)[-(top_k + 1):][::-1]

        return [self.idx2word[idx] for idx in top_indices]

    def plot_clusters(self, seed_words, top_k=15, use_tsne=True):
        print("\nExtracting clusters for visualization...")

        words_to_plot = []
        cluster_labels = []

        # Gather the seed words and their neighbors
        for idx, seed in enumerate(seed_words):
            cluster = self.get_similar_words(seed, top_k)
            if not cluster:
                print(f"Word '{seed}' not found. Skipping.")
                continue

            words_to_plot.extend(cluster)
            cluster_labels.extend([idx] * len(cluster))

        # Get the 100D vectors for the selected words
        indices = [self.word2idx[w] for w in words_to_plot]
        vectors = self.normalized_embeddings[indices]

        print("Reducing dimensions to 2D...")
        if use_tsne:
            reducer = TSNE(n_components=2, perplexity=min(30, len(words_to_plot) - 1), random_state=42)
        else:
            reducer = PCA(n_components=2)

        vectors_2d = reducer.fit_transform(vectors)

        plt.figure(figsize=(14, 10))
        colors = plt.get_cmap('tab10', len(seed_words))

        texts = []

        for i, word in enumerate(words_to_plot):
            x, y = vectors_2d[i, 0], vectors_2d[i, 1]
            cluster_idx = cluster_labels[i]

            is_seed = word in seed_words
            size = 100 if is_seed else 40
            alpha = 1.0 if is_seed else 0.7

            plt.scatter(x, y, color=colors(cluster_idx), s=size, alpha=alpha, edgecolors='w', linewidth=0.5)

            # Create the text object
            texts.append(plt.text(
                x, y, word,
                fontsize=13 if is_seed else 10,
                fontweight='bold' if is_seed else 'normal',
                color='black' if is_seed else '#333333'
            ))

        title = "t-SNE" if use_tsne else "PCA"
        plt.title(f"Word2Vec Embeddings ({title} Projection)", fontsize=18, pad=15)
        plt.grid(True, linestyle='--', alpha=0.3)

        print("Adjusting text labels to prevent overlap (this takes a second)...")
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.6),
            expand_points=(1.2, 1.2),
            expand_text=(1.2, 1.2)
        )

        plt.tight_layout()
        save_path = f"../final_models/cluster_plot_{title.lower()}_adjusted.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.show()


if __name__ == "__main__":
    try:
        visualizer = Word2VecVisualizer()

        # Few distinct concepts to see them group up
        words = ["computer", "king", "water", "math", "europe"]

        visualizer.plot_clusters(words, top_k=15, use_tsne=True)

    except FileNotFoundError as e:
        print(e)
