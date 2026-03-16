import numpy as np

def sigmoid(x):
    """Numerically stable sigmoid function to prevent overflow."""
    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))


class Word2Vec:
    """
    Skip-Gram with Negative Sampling (SGNS) model implemented in NumPy.
    """

    def __init__(self, vocab_size, embedding_dim=100, learning_rate=0.025, l2_reg=1e-4):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg

        # W1: Center word embeddings (v_c)
        self.W1 = np.random.uniform(
            -0.5 / embedding_dim, 0.5 / embedding_dim,
            (vocab_size, embedding_dim)
        )

        # W2: Context word embeddings (u_o and u_k)
        self.W2 = np.zeros((vocab_size, embedding_dim))

    def update(self, center_words, context_words, negative_samples):
        """
        Performs the forward/backward pass using optimized Einstein Summation (einsum)
        to prevent massive memory bandwidth bottlenecks.
        """
        # Fetch Embeddings
        v_c = self.W1[center_words]  # (Batch, Dim)
        u_o = self.W2[context_words]  # (Batch, Dim)
        u_k = self.W2[negative_samples]  # (Batch, Num_Neg, Dim)

        # 2. Forward Pass (einsum)
        # Dot product of corresponding rows: (Batch, Dim) * (Batch, Dim) -> (Batch,)
        z_pos = np.einsum('bd,bd->b', v_c, u_o)

        # Batch matrix multiplication: (Batch, Num_Neg, Dim) * (Batch, Dim) -> (Batch, Num_Neg)
        z_neg = np.einsum('bnd,bd->bn', u_k, v_c)

        loss = -np.sum(np.log(sigmoid(z_pos))) - np.sum(np.log(sigmoid(-z_neg)))

        # (lambda / 2) * sum(weights^2)
        l2_loss = 0.5 * self.l2_reg * (float(np.sum(v_c ** 2)) + float(np.sum(u_o ** 2)) + float(np.sum(u_k ** 2)))
        loss += l2_loss

        # Compute Prediction Errors
        e_pos = sigmoid(z_pos) - 1.0  # (Batch,)
        e_neg = sigmoid(z_neg)  # (Batch, Num_Neg)

        # Backpropagation
        # Using standard broadcasting instead of einsum
        grad_v_c = (e_pos[:, None] * u_o) + np.sum(e_neg[:, :, None] * u_k, axis=1)
        grad_u_o = e_pos[:, None] * v_c
        grad_u_k = e_neg[:, :, None] * v_c[:, None, :]

        # L2 Regularization
        grad_v_c += self.l2_reg * v_c
        grad_u_o += self.l2_reg * u_o
        grad_u_k += self.l2_reg * u_k

        # Gradient clipping
        clip_val = 1.0
        grad_v_c = np.clip(grad_v_c, -clip_val, clip_val)
        grad_u_o = np.clip(grad_u_o, -clip_val, clip_val)
        grad_u_k = np.clip(grad_u_k, -clip_val, clip_val)

        # Parameter Updates
        np.add.at(self.W1, center_words, -self.learning_rate * grad_v_c)
        np.add.at(self.W2, context_words, -self.learning_rate * grad_u_o)

        np.add.at(self.W2, negative_samples.ravel(), -self.learning_rate * grad_u_k.reshape(-1, self.embedding_dim))

        return loss
