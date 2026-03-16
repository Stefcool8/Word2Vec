import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for float32 arrays."""
    x = np.clip(x, -15.0, 15.0)
    return 1.0 / (1.0 + np.exp(-x))


class Word2Vec:
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        learning_rate: float = 0.025,
        l2_reg: float = 1e-4,
        clip_val: float = None,
        dtype=np.float32,
    ):
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.learning_rate = float(learning_rate)
        self.l2_reg = float(l2_reg)
        self.clip_val = clip_val
        self.dtype = dtype

        # initialization scale similar to original word2vec prescriptions
        scale = 0.5 / float(self.embedding_dim)

        # center embeddings (v_c)
        self.W1 = (
            (np.random.rand(self.vocab_size, self.embedding_dim).astype(self.dtype) - 0.5)
            * 2.0
            * scale
        )

        # context embeddings (u_o and u_k) - random, not zero
        self.W2 = (
            (np.random.rand(self.vocab_size, self.embedding_dim).astype(self.dtype) - 0.5)
            * 2.0
            * scale
        )

    def update(self, center_words, context_words, negative_samples):
        """
        Perform forward and backward pass and update parameters.

        Parameters
        ----------
        center_words : array-like, shape (B,)
            indices of center words (int)
        context_words : array-like, shape (B,)
            indices of positive context words (int)
        negative_samples : array-like, shape (B, N)
            indices of negative samples per center (int)

        Returns
        -------
        avg_loss : float
            average loss per positive pair (useful for logging)
        """
        # ensure dtypes
        center_words = np.asarray(center_words, dtype=np.int32)
        context_words = np.asarray(context_words, dtype=np.int32)
        negative_samples = np.asarray(negative_samples, dtype=np.int32)

        B = center_words.shape[0]
        if B == 0:
            return 0.0
        N = negative_samples.shape[1]

        # 1) fetch embeddings
        v_c = self.W1[center_words]         # (B, D), view into W1
        u_o = self.W2[context_words]        # (B, D)
        u_k = self.W2[negative_samples]     # (B, N, D)

        # forward logits
        z_pos = np.einsum('bd,bd->b', v_c, u_o).astype(self.dtype)    # (B,)
        z_neg = np.einsum('bnd,bd->bn', u_k, v_c).astype(self.dtype)  # (B, N)

        eps = 1e-10
        pos_loss = -np.log(sigmoid(z_pos) + eps).sum()
        neg_loss = -np.log(sigmoid(-z_neg) + eps).sum()
        loss = pos_loss + neg_loss

        # L2 regularization added to scalar loss (optional)
        if self.l2_reg > 0.0:
            l2_loss = 0.5 * self.l2_reg * (np.sum(v_c * v_c) + np.sum(u_o * u_o) + np.sum(u_k * u_k))
            loss += l2_loss

        # prediction errors
        e_pos = (sigmoid(z_pos) - 1.0).astype(self.dtype)   # (B,)
        e_neg = sigmoid(z_neg).astype(self.dtype)           # (B, N)

        # per-example gradients
        # grad w.r.t center vectors: shape (B, D)
        grad_v_c = (e_pos[:, None] * u_o) + np.sum(e_neg[:, :, None] * u_k, axis=1)
        grad_u_o = (e_pos[:, None] * v_c)                    # (B, D)
        grad_u_k = (e_neg[:, :, None] * v_c[:, None, :])    # (B, N, D)

        # add L2 gradients for those sampled embeddings (small pull to zero)
        if self.l2_reg > 0.0:
            grad_v_c += self.l2_reg * v_c
            grad_u_o += self.l2_reg * u_o
            grad_u_k += self.l2_reg * u_k

        # aggregate gradients by unique indices
        # centers (W1)
        unique_centers, inv_centers = np.unique(center_words, return_inverse=True)
        agg_grad_v = np.zeros((unique_centers.shape[0], self.embedding_dim), dtype=self.dtype)
        np.add.at(agg_grad_v, inv_centers, grad_v_c)

        # compute counts per unique center index and average per-occurrence
        counts_centers = np.bincount(inv_centers, minlength=unique_centers.shape[0]).astype(self.dtype)
        # avoid dividing by zero (shouldn't happen)
        counts_centers[counts_centers == 0] = 1.0
        agg_grad_v /= counts_centers[:, None]   # mean gradient per occurrence for that word

        # positive contexts (W2)
        unique_ctx, inv_ctx = np.unique(context_words, return_inverse=True)
        agg_grad_u_o = np.zeros((unique_ctx.shape[0], self.embedding_dim), dtype=self.dtype)
        np.add.at(agg_grad_u_o, inv_ctx, grad_u_o)
        counts_ctx = np.bincount(inv_ctx, minlength=unique_ctx.shape[0]).astype(self.dtype)
        counts_ctx[counts_ctx == 0] = 1.0
        agg_grad_u_o /= counts_ctx[:, None]

        # negatives (W2) - flatten for unique aggregation
        neg_flat = negative_samples.ravel()
        grad_u_k_flat = grad_u_k.reshape(-1, self.embedding_dim)
        unique_neg, inv_neg = np.unique(neg_flat, return_inverse=True)
        agg_grad_u_k = np.zeros((unique_neg.shape[0], self.embedding_dim), dtype=self.dtype)
        np.add.at(agg_grad_u_k, inv_neg, grad_u_k_flat)
        counts_neg = np.bincount(inv_neg, minlength=unique_neg.shape[0]).astype(self.dtype)
        counts_neg[counts_neg == 0] = 1.0
        agg_grad_u_k /= counts_neg[:, None]

        # optional gradient clipping on aggregated arrays
        if self.clip_val is not None:
            np.clip(agg_grad_v, -self.clip_val, self.clip_val, out=agg_grad_v)
            np.clip(agg_grad_u_o, -self.clip_val, self.clip_val, out=agg_grad_u_o)
            np.clip(agg_grad_u_k, -self.clip_val, self.clip_val, out=agg_grad_u_k)

        # parameter updates (single indexed updates
        lr = self.learning_rate
        # W1 update
        self.W1[unique_centers] -= lr * agg_grad_v
        # W2 updates for positive contexts
        self.W2[unique_ctx] -= lr * agg_grad_u_o
        # W2 updates for negative samples
        self.W2[unique_neg] -= lr * agg_grad_u_k

        avg_loss = float(loss) / float(B)
        return avg_loss