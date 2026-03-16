# Pure NumPy Word2Vec (Skip-Gram with Negative Sampling)

A from-scratch, highly optimized implementation of the Word2Vec algorithm (Skip-Gram variant) using **100% pure Python and NumPy**. No PyTorch, TensorFlow, or other deep learning frameworks were used.

This project demonstrates a deep understanding of natural language processing mathematics, matrix calculus, and memory-efficient array manipulation in Python.

## Features

- **Strictly Pure NumPy**: Forward passes, backpropagation, and parameter updates are all derived and calculated using pure NumPy matrix operations.

- **Skip-Gram with Negative Sampling (SGNS)**: Implements the highly efficient negative sampling architecture to approximate the softmax denominator.

- **Vectorized Data Processing**: Context window slicing and subsampling masks are fully vectorized in C-level NumPy, completely avoiding slow Python ```for``` loops during pair generation.

- **Frequent Word Subsampling**: Implements Mikolov's probabilistic subsampling threshold to balance the learning between rare and highly frequent words.

- **Mathematical Stability**: Includes pre-emptive gradient clipping and numerically stable sigmoid functions to prevent floating-point explosions (NaNs).

- **Semantic Evaluation Suite**: Includes scripts to track the real-time evolution of classic word analogies (e.g., *king - man + woman = queen*) across training epochs.

## Note on Performance & CPU Utilization

If you monitor system resources during training, you will notice that CPU utilization remains relatively low (often pegged to a single core). **This is a known, expected limitation of strict pure-NumPy Word2Vec implementations**.

To maintain perfect mathematical correctness without using a compiled C++ extension or Cython:

1. Frequent words (like "the") appear hundreds of times in a single batch. To prevent race conditions and ensure every gradient update is applied correctly, we must use NumPy's ```np.add.at()```. This function forces a strict, sequential, single-threaded C loop.

2. Batched Matmul Limitations: While NumPy's underlying BLAS backend (OpenBLAS/MKL) excels at multithreading massive 2D matrix multiplications, it does not automatically multithread the thousands of tiny batched 1D/2D dot products required for negative sampling.

Production libraries (like Gensim) achieve 100% CPU utilization by dropping out of pure Python and using Cython to release the Global Interpreter Lock (GIL) and perform lock-free atomic updates (Hogwild! training). **This codebase prioritizes pure mathematical correctness and strict adherence to the NumPy constraint over unsafe multithreaded workarounds**.

## Project Structure

```
├── data/
│   └── raw/
│       └── text8                # The text8 corpus (downloaded separately)
├── saved_models/                # Auto-generated directory for .npy weights and vocab
├── src/
│   ├── data_loader.py           # Vectorized pair generation, subsampling, unigram table
│   ├── word2vec.py              # Core pure-NumPy model, forward/backward pass logic
│   ├── evaluate.py              # Cosine similarity and analogy resolution tool
│   └── track_evolution.py       # Tracks analogy progress across saved epoch checkpoints             
└── main.py                      # Training loop and dynamic learning rate decay
```

## Installation & Setup

### 1. Clone the repository:
```
git clone [https://github.com/yourusername/numpy-word2vec.git](https://github.com/yourusername/numpy-word2vec.git)
cd numpy-word2vec
```

### 2. Install dependencies: <span style="font-weight:normal;">Only ```numpy``` is required.</span>
```
pip install numpy
```

### 3. Download the Dataset:
Place the [text8 corpus](http://mattmahoney.net/dc/text8.zip) (or any clean text file) into ```data/raw/text8```.

## Usage

### 1. Train the Model

Run the main script to start data preparation and training. The model automatically saves matrix weights (```W1```, ```W2```) and the vocabulary at the end of each epoch.
```
python main.py
```

### 2. Evaluate Semantic Similarities

Once you have trained the model (or after the first epoch finishes), you can test nearest neighbors and analogies.
```
python evaluate.py
```

#### Example Output (hypothetical analogy for now; training in progress...)
```
Analogy Test: king - man + woman = ?
========================================
  queen: 0.6821
  monarch: 0.5432
  princess: 0.5110
```

### 3. Track Analogy Evolution

Watch how the model learns a specific semantic relationship over time by tracking it across all saved epoch checkpoints.
```
python track_evolution.py
```

## Core Hyperparameters (<span style="font-size:16px; font-weight:normal;">```main.py```</span>)

- ```EMBEDDING_DIM = 100```

- ```WINDOW_SIZE = 5```

- ```NUM_NEG_SAMPLES = 15```

- ```INITIAL_LEARNING_RATE = 0.025```

- ```BATCH_SIZE = 2048``` (*Can be scaled up depending on RAM*)