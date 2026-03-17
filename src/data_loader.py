from collections import Counter

import numpy as np


class Word2VecDataLoader:
    """
    Handles reading text data, building vocabulary, subsampling frequent words,
    and generating training batches for a Word2Vec (Skip-Gram) model.
    """

    def __init__(self, file_path, min_count=5, window_size=5, sample_threshold=1e-3):
        self.file_path = file_path
        self.min_count = min_count
        self.window_size = window_size
        self.sample_threshold = sample_threshold

        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.corpus_indices = []
        self.neg_sample_probs = None

        self.context_words_array = None
        self.center_words_array = None

        print("Initializing DataLoader and building vocabulary...")
        self._build_vocab()
        self._subsample_corpus()
        self._init_negative_sampling_distribution()

    def _build_vocab(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        tokens = text.split()
        word_counts = Counter(tokens)
        vocab_words = [word for word, count in word_counts.items() if count >= self.min_count]
        self.vocab_size = len(vocab_words)

        self.word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab_words)}
        self.corpus_indices = [self.word2idx[w] for w in tokens if w in self.word2idx]
        self.word_counts_array = np.array([word_counts[self.idx2word[i]] for i in range(self.vocab_size)])

        print(f"Vocabulary built: {self.vocab_size:,} unique words.")
        print(f"Original corpus size (after min_count): {len(self.corpus_indices):,} words.")

    def _subsample_corpus(self):
        print(f"Applying subsampling (threshold = {self.sample_threshold})...")
        total_words = np.sum(self.word_counts_array)
        frequencies = self.word_counts_array / total_words

        # Subsampling probability formula
        keep_probs = (np.sqrt(frequencies / self.sample_threshold) + 1) * (self.sample_threshold / frequencies)
        keep_probs = np.clip(keep_probs, 0.0, 1.0)

        corpus = np.array(self.corpus_indices, dtype=np.int32)
        token_probs = keep_probs[corpus]
        random_values = np.random.rand(len(corpus))
        keep_mask = random_values < token_probs
        self.corpus_indices = corpus[keep_mask].tolist()

        print(f"Subsampling complete. Corpus size reduced to {len(self.corpus_indices):,} words.")

    def _init_negative_sampling_distribution(self):
        # 3/4 power empirically dampens the frequency of highly common words
        powered_counts = self.word_counts_array ** 0.75
        self.neg_sample_probs = powered_counts / np.sum(powered_counts)
        self.table_size = int(1e8)
        self.unigram_table = np.zeros(self.table_size, dtype=np.int32)

        curr_idx = 0
        for word_idx, prob in enumerate(self.neg_sample_probs):
            count = int(prob * self.table_size)
            self.unigram_table[curr_idx: curr_idx + count] = word_idx
            curr_idx += count

        if curr_idx < self.table_size:
            self.unigram_table[curr_idx:] = self.vocab_size - 1

    def get_negative_samples(self, batch_size, num_samples):
        # Fast table lookup for negative sampling
        random_indices = np.random.randint(0, self.table_size, size=(batch_size, num_samples))
        return self.unigram_table[random_indices]

    def prepare_data(self):
        """
        Pre-computes pairs using highly optimized vectorized slicing.
        Applies a probabilistic mask to mathematically replicate dynamic window sizing.
        """
        print("Generating training pairs via vectorized slices...")
        corpus = np.array(self.corpus_indices, dtype=np.int32)

        all_centers = []
        all_contexts = []

        # Vectorized dynamic window size
        for offset in range(1, self.window_size + 1):
            # Probability of keeping a word at this distance
            keep_prob = (self.window_size - offset + 1) / self.window_size

            # Forward context
            centers_fwd = corpus[:-offset]
            contexts_fwd = corpus[offset:]
            mask_fwd = np.random.rand(len(centers_fwd)) < keep_prob

            all_centers.append(centers_fwd[mask_fwd])
            all_contexts.append(contexts_fwd[mask_fwd])

            # Backward context
            centers_bwd = corpus[offset:]
            contexts_bwd = corpus[:-offset]
            mask_bwd = np.random.rand(len(centers_bwd)) < keep_prob

            all_centers.append(centers_bwd[mask_bwd])
            all_contexts.append(contexts_bwd[mask_bwd])

        self.center_words_array = np.concatenate(all_centers)
        self.context_words_array = np.concatenate(all_contexts)

        print(f"Generated {len(self.center_words_array):,} training pairs.")

    def generate_batches(self, batch_size):
        """
        Shuffles the dataset at the start of each epoch and yields fast slices.
        """
        num_pairs = len(self.center_words_array)

        # Epoch-level shuffling
        # This fixes the stagnation spike at the beginning of new epochs
        shuffle_indices = np.random.permutation(num_pairs)
        self.center_words_array = self.center_words_array[shuffle_indices]
        self.context_words_array = self.context_words_array[shuffle_indices]

        for i in range(0, num_pairs, batch_size):
            yield (
                self.center_words_array[i: i + batch_size],
                self.context_words_array[i: i + batch_size]
            )
