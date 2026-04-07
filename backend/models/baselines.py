"""
Baseline Models for comparison with the Two-Tower Neural Network.

1. PopularityBaseline: recommends the most frequently procured products
2. MatrixFactorizationALS: collaborative filtering via ALS on implicit feedback
"""

import numpy as np
from scipy.sparse import csr_matrix


class PopularityBaseline:
    """Non-personalized baseline: ranks products by global procurement frequency."""

    def __init__(self):
        self.product_scores = None
        self.n_products = 0

    def fit(self, interactions, n_products):
        self.n_products = n_products
        self.product_scores = np.zeros(n_products)
        for _, pid, _ in interactions:
            self.product_scores[pid] += 1
        # Normalize to probabilities
        total = self.product_scores.sum()
        if total > 0:
            self.product_scores /= total

    def recommend(self, buyer_id, k=10, exclude=None):
        scores = self.product_scores.copy()
        if exclude:
            for pid in exclude:
                if pid < len(scores):
                    scores[pid] = -1
        top_k = np.argsort(scores)[::-1][:k]
        return top_k.tolist()

    def get_scores(self, buyer_id):
        return self.product_scores.copy()


class MatrixFactorizationALS:
    """
    Matrix Factorization with Alternating Least Squares for implicit feedback.
    Based on Hu, Koren, and Volinsky (2008).
    """

    def __init__(self, n_factors=64, regularization=0.01, alpha=40,
                 n_iterations=15):
        self.n_factors = n_factors
        self.regularization = regularization
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.user_factors = None
        self.item_factors = None

    def fit(self, interactions, n_buyers, n_products):
        # Build confidence matrix C = 1 + alpha * R
        rows, cols, data = [], [], []
        for bid, pid, _ in interactions:
            rows.append(bid)
            cols.append(pid)
            data.append(1.0)

        R = csr_matrix((data, (rows, cols)), shape=(n_buyers, n_products))
        C = R.multiply(self.alpha) + csr_matrix(np.ones((n_buyers, n_products)))

        # Initialize factors randomly
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.01, (n_buyers, self.n_factors))
        self.item_factors = np.random.normal(0, 0.01, (n_products, self.n_factors))

        lambda_I = self.regularization * np.eye(self.n_factors)

        print("Training Matrix Factorization (ALS)...")
        for iteration in range(self.n_iterations):
            # Fix items, solve for users
            YtY = self.item_factors.T @ self.item_factors
            for u in range(n_buyers):
                # Get non-zero entries for this user
                cu = np.array(C[u].todense()).flatten()
                pu = (R[u].toarray().flatten() > 0).astype(float)

                # Cu - I diagonal
                cu_diag = np.diag(cu)

                # Solve: (Y^T C_u Y + lambda*I) x_u = Y^T C_u p_u
                A = YtY + self.item_factors.T @ (cu_diag - np.eye(n_products)) @ self.item_factors + lambda_I
                b = self.item_factors.T @ cu_diag @ pu
                self.user_factors[u] = np.linalg.solve(A, b)

            # Fix users, solve for items
            XtX = self.user_factors.T @ self.user_factors
            for i in range(n_products):
                ci = np.array(C[:, i].todense()).flatten()
                pi = (R[:, i].toarray().flatten() > 0).astype(float)

                ci_diag = np.diag(ci)

                A = XtX + self.user_factors.T @ (ci_diag - np.eye(n_buyers)) @ self.user_factors + lambda_I
                b = self.user_factors.T @ ci_diag @ pi
                self.item_factors[i] = np.linalg.solve(A, b)

            # Compute loss for monitoring
            if (iteration + 1) % 5 == 0:
                pred = self.user_factors @ self.item_factors.T
                R_dense = R.toarray()
                C_dense = C.toarray()
                weighted_error = np.sum(C_dense * (R_dense - pred) ** 2)
                reg_term = self.regularization * (
                    np.sum(self.user_factors ** 2) + np.sum(self.item_factors ** 2)
                )
                loss = weighted_error + reg_term
                print(f"  Iteration {iteration + 1}/{self.n_iterations}, Loss: {loss:.4f}")

        print("  MF-ALS training complete.")

    def recommend(self, buyer_id, k=10, exclude=None):
        if self.user_factors is None:
            return []
        scores = self.user_factors[buyer_id] @ self.item_factors.T
        if exclude:
            for pid in exclude:
                if pid < len(scores):
                    scores[pid] = -np.inf
        top_k = np.argsort(scores)[::-1][:k]
        return top_k.tolist()

    def get_scores(self, buyer_id):
        if self.user_factors is None:
            return np.zeros(self.item_factors.shape[0])
        return self.user_factors[buyer_id] @ self.item_factors.T
