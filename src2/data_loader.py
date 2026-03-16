from collections import Counter

import numpy as np


class Word2VecDataLoader:
    """
    Handles reading text data, building vocabulary, subsampling frequent words,
    and generating training batches for a Word2Vec (Skip-Gram) model.

    - This implementation samples negatives with numpy.random.choice
      using the 0.75-power unigram distribution. This avoids constructing a
      huge unigram table (which uses a lot of memory).
    """

    def __init__(
        self,
        file_path,
        min_count=5,
        window_size=5,
        sample_threshold=1e-3,
        use_unigram_table: bool = False,
        unigram_table_size: int = int(1e7),
    ):
        self.file_path = file_path
        self.min_count = min_count
        self.window_size = window_size
        self.sample_threshold = sample_threshold

        # negative sampling configuration
        self.use_unigram_table = bool(use_unigram_table)
        self.table_size = int(unigram_table_size)

        # placeholders
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

        # map words to indices; using enumerate keeps a stable mapping order
        self.word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab_words)}

        # build corpus indices (only words that passed min_count)
        self.corpus_indices = [self.word2idx[w] for w in tokens if w in self.word2idx]

        # counts as numpy array (int32 is enough for counts up to billions in practice)
        # use float64 when needed for probability math
        self.word_counts_array = np.array([word_counts[self.idx2word[i]] for i in range(self.vocab_size)], dtype=np.int64)

        print(f"Vocabulary built: {self.vocab_size:,} unique words.")
        print(f"Original corpus size (after min_count): {len(self.corpus_indices):,} words.")

    def _subsample_corpus(self):
        """
        Subsampling frequent words (Mikolov et al.). Keeps tokens with probability
        defined by the sampling formula. Operates on index-level arrays and
        stores back to self.corpus_indices as a list of int32.
        """
        print(f"Applying subsampling (threshold = {self.sample_threshold})...")
        total_words = float(np.sum(self.word_counts_array))
        # avoid division by zero
        if total_words <= 0.0:
            return

        frequencies = self.word_counts_array.astype(np.float64) / total_words
        # Mikolov sampling formula
        keep_probs = (np.sqrt(frequencies / self.sample_threshold) + 1.0) * (self.sample_threshold / (frequencies + 1e-12))
        keep_probs = np.clip(keep_probs, 0.0, 1.0)

        corpus = np.array(self.corpus_indices, dtype=np.int32)
        token_probs = keep_probs[corpus]
        random_values = np.random.rand(corpus.shape[0])
        keep_mask = random_values < token_probs
        self.corpus_indices = corpus[keep_mask].astype(np.int32).tolist()

        print(f"Subsampling complete. Corpus size reduced to {len(self.corpus_indices):,} words.")

    def _init_negative_sampling_distribution(self):
        """
        Prepares the negative sampling distribution (power 0.75).
        Optionally prepares a unigram table if requested (smaller table_size).
        """
        # 0.75 power distribution
        powered = np.power(self.word_counts_array.astype(np.float64), 0.75)
        total_powered = powered.sum()
        if total_powered <= 0:
            # safety fallback to uniform
            self.neg_sample_probs = np.ones(self.vocab_size, dtype=np.float64) / float(self.vocab_size)
        else:
            self.neg_sample_probs = (powered / total_powered).astype(np.float64)

        # only build the unigram table if explicitly requested
        if self.use_unigram_table:
            print(f"Building unigram table of size {self.table_size:,} (this may take memory)...")
            self.unigram_table = np.zeros(self.table_size, dtype=np.int32)
            curr_idx = 0
            for word_idx, prob in enumerate(self.neg_sample_probs):
                # allocate approx prob * table_size slots
                count = int(prob * float(self.table_size))
                if count <= 0:
                    continue
                end = min(self.table_size, curr_idx + count)
                self.unigram_table[curr_idx:end] = word_idx
                curr_idx = end
                if curr_idx >= self.table_size:
                    break
            if curr_idx < self.table_size:
                # fill remaining with last index to avoid uninitialized area
                self.unigram_table[curr_idx:] = self.vocab_size - 1
            print("Unigram table built.")

    def get_negative_samples(self, batch_size, num_samples):
        """
        Returns shape (batch_size, num_samples) of negative sample indices (int32).

        Two modes:
         - If use_unigram_table: sample from the prebuilt unigram_table using randint
         - Else: use np.random.choice with the probability distribution neg_sample_probs
        """
        if batch_size <= 0 or num_samples <= 0:
            return np.empty((0, 0), dtype=np.int32)

        if self.use_unigram_table:
            # fast integer indexing into a prebuilt table
            random_indices = np.random.randint(0, self.table_size, size=(batch_size, num_samples))
            return self.unigram_table[random_indices].astype(np.int32)
        else:
            # sample directly from the distribution; this is memory-light and simpler
            # numpy's choice with p= is implemented in C and is reasonably fast for moderate sizes
            # generate flattened then reshape to avoid one big allocation in some numpy versions
            flat = np.random.choice(self.vocab_size, size=(batch_size * num_samples), p=self.neg_sample_probs)
            return flat.reshape(batch_size, num_samples).astype(np.int32)

    def prepare_data(self):
        """
        Pre-computes pairs using vectorized slicing across dynamic windows.
        Stores the resulting arrays as np.int32 for efficiency.
        """
        print("Generating training pairs via vectorized slices...")
        corpus = np.array(self.corpus_indices, dtype=np.int32)

        all_centers = []
        all_contexts = []

        # dynamic window: for each offset include forward and backward contexts
        for offset in range(1, self.window_size + 1):
            keep_prob = float(self.window_size - offset + 1) / float(self.window_size)

            # forward context
            centers_fwd = corpus[:-offset]
            contexts_fwd = corpus[offset:]
            mask_fwd = np.random.rand(centers_fwd.shape[0]) < keep_prob
            if mask_fwd.any():
                all_centers.append(centers_fwd[mask_fwd])
                all_contexts.append(contexts_fwd[mask_fwd])

            # backward context
            centers_bwd = corpus[offset:]
            contexts_bwd = corpus[:-offset]
            mask_bwd = np.random.rand(centers_bwd.shape[0]) < keep_prob
            if mask_bwd.any():
                all_centers.append(centers_bwd[mask_bwd])
                all_contexts.append(contexts_bwd[mask_bwd])

        if not all_centers:
            # fallback in degenerate case
            self.center_words_array = np.array([], dtype=np.int32)
            self.context_words_array = np.array([], dtype=np.int32)
            print("No training pairs generated.")
            return

        self.center_words_array = np.concatenate(all_centers).astype(np.int32)
        self.context_words_array = np.concatenate(all_contexts).astype(np.int32)

        num_pairs = len(self.center_words_array)
        print(f"Generated {num_pairs:,} training pairs. Shuffling...")

        # shuffle in unison
        shuffle_indices = np.random.permutation(num_pairs)
        self.center_words_array = self.center_words_array[shuffle_indices]
        self.context_words_array = self.context_words_array[shuffle_indices]

    def generate_batches(self, batch_size):
        """Yields slices from the pre-computed arrays (center, context) as int32 arrays."""
        num_pairs = len(self.center_words_array)
        for i in range(0, num_pairs, batch_size):
            yield (
                self.center_words_array[i: i + batch_size],
                self.context_words_array[i: i + batch_size]
            )
