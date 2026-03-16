import numpy as np


def sigmoid(x):
    """Numerically stable sigmoid function to prevent overflow."""
    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))


class Word2Vec:
    """
    Skip-Gram with Negative Sampling (SGNS) model implemented in pure C-level NumPy.
    """

    def __init__(self, vocab_size, embedding_dim=100, learning_rate=0.025):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        # W1: Center word embeddings (v_c)
        self.W1 = np.random.uniform(
            -0.5 / embedding_dim, 0.5 / embedding_dim,
            (vocab_size, embedding_dim)
        )

        # W2: Context word embeddings (u_o and u_k)
        self.W2 = np.zeros((vocab_size, embedding_dim))

    def update(self, center_words, context_words, negative_samples):
        """
        Performs the forward/backward pass using optimized BLAS matrix multiplications
        instead of memory-heavy 3D broadcasting or slow einsum loops.
        """
        # Fetch Embeddings
        v_c = self.W1[center_words]  # (Batch, Dim)
        u_o = self.W2[context_words]  # (Batch, Dim)
        u_k = self.W2[negative_samples]  # (Batch, Num_Neg, Dim)

        # Forward Pass (Optimized BLAS Matmul)
        # Fast batched dot product for positive samples: (Batch,)
        z_pos = np.sum(v_c * u_o, axis=1)

        # Batched matrix multiplication for negative samples:
        # (Batch, Num_Neg, Dim) @ (Batch, Dim, 1) -> (Batch, Num_Neg, 1) -> (Batch, Num_Neg)
        z_neg = (u_k @ v_c[:, :, None]).squeeze(-1)

        # Pure semantic loss (L2 regularization removed for proper convergence)
        loss = -np.sum(np.log(sigmoid(z_pos))) - np.sum(np.log(sigmoid(-z_neg)))

        # Compute Prediction Errors
        e_pos = sigmoid(z_pos) - 1.0  # (Batch,)
        e_neg = sigmoid(z_neg)  # (Batch, Num_Neg)

        # Backpropagation
        # Fast batched matrix multiplication.
        # Shape trace: (Batch, 1, Num_Neg) @ (Batch, Num_Neg, Dim) -> (Batch, 1, Dim) -> (Batch, Dim)
        grad_v_c = (e_pos[:, None] * u_o) + (e_neg[:, None, :] @ u_k).squeeze(1)
        grad_u_o = e_pos[:, None] * v_c
        grad_u_k = e_neg[:, :, None] * v_c[:, None, :]

        # Parameter Updates
        # Using sparse updates directly into the weight matrices. No clipping
        np.add.at(self.W1, center_words, -self.learning_rate * grad_v_c)
        np.add.at(self.W2, context_words, -self.learning_rate * grad_u_o)

        # Flatten the negative samples and reshape the gradients to match for bulk update
        np.add.at(self.W2, negative_samples.ravel(), -self.learning_rate * grad_u_k.reshape(-1, self.embedding_dim))

        return loss
