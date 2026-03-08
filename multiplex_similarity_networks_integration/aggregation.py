import numpy as np
from scipy import linalg
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from utils import *

class SimilarityMatrixAggregator(ABC):
    """Abstract base class for our aggregation (of similarity matrices) methods."""

    def __init__(self, matrices: List[np.ndarray], weights: Optional[np.ndarray] = None):
        self.matrices = matrices
        self.weights, self.weights_source, self.RV_matrix = self._resolve_weights(weights)
        self._validate_input()

    def _validate_input(self) -> None:
        """Validate the inputs."""
        if not self.matrices:
            raise ValueError("The list of matrices cannot be empty")

        # Check if all the matrices are numpy array 2D
        for i, matrix in enumerate(self.matrices):
            if not isinstance(matrix, np.ndarray):
                raise TypeError(f"Matrix {i} is not a numpy array")
            if matrix.ndim != 2:
                raise ValueError(f"Matrix {i} is not 2D")
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"Matrix {i} is not square")

        # Check if the dimensions are consistent
        first_shape = self.matrices[0].shape
        for i, matrix in enumerate(self.matrices[1:]):
            if matrix.shape != first_shape:
                raise ValueError(f"Matrix {i+1} has different dimensions from the first one")

    def _resolve_weights(self, weights: Optional[np.ndarray]) -> Tuple[np.ndarray, str, Optional[np.ndarray]]:
        """Manage the weights: if given they get validated, otherwise they get computed.
        In particular, the function returns (weights, source, RV matrix)."""
        if weights is not None:
            # Weights validation if provided as input
            if len(weights) != len(self.matrices):
                raise ValueError(f"Number of weights ({len(weights)}) != number of matrices ({len(self.matrices)})")
            if not np.all(weights >= 0):
                raise ValueError("All the weights must be non-negative")
            if np.sum(weights) == 0:
                raise ValueError("The sum of the weights cannot be zero")

            normalized_weights = weights / np.sum(weights)  # Normalize
            return normalized_weights, "provided", None  # If the weights are provided, there's no RV matrix
        else:
            # Compute the weights based on the aggregation method chosen
            computed_weights, RV_matrix = self._compute_method_specific_weights()
            return computed_weights, "computed", RV_matrix

    # We use the decorator @abstractmethod to highlight the methods (abstract methods)
    # which must be implemented by subclasses.
    @abstractmethod
    def _compute_method_specific_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def aggregate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute the aggregation matrix."""
        pass

    # The following method returns the weights used for the aggregation:
    # we obtain an array NumPy or None if the weights are not available (type Optional
    # is used as a precaution).
    def get_weights(self) -> Optional[np.ndarray]:
        """Return the weights used for the aggregation."""
        return self.weights


class WeightedMeanAggregator(SimilarityMatrixAggregator):
    """Aggregator for the (Frobenius) weighted arithmetic mean."""

    def __init__(self, matrices: List[np.ndarray], weights: Optional[np.ndarray] = None):
        super().__init__(matrices, weights)  # The matrices and the weights are managed by the base (abstract) class 
        self.weight_evaluation = eval_weights(self.RV_matrix)  # Function that evaluates how well the weights represent the matrices
    def _compute_method_specific_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the weights (and the RV matrix) using the Frobenius method."""
        # Create the RV matrix to obtain similarity values between matrices
        n = len(self.matrices)
        RV = np.identity(n)

        for i in range(n-1):
            for j in range(i+1, n):
                RV[i, j] = np.trace(self.matrices[i].T @ self.matrices[j]) / (
                    np.linalg.norm(self.matrices[i], 'fro') * np.linalg.norm(self.matrices[j], 'fro'))

        RV = RV + RV.T - np.identity(n)
        weights = frobenius_weights(RV)
        return weights, RV

    def aggregate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        weighted_sum = np.zeros_like(self.matrices[0], dtype=np.float64)
        for w, matrix in zip(self.weights, self.matrices):
            weighted_sum += w * matrix

        info = {
            "method": "weighted_arithmetic_mean",
            "weights": self.weights.copy(),
            "RV_matrix": self.RV_matrix,  # RV matrix
            "weight_evaluation": self.weight_evaluation 
        }
        return weighted_sum, info

class GeometricAggregator(SimilarityMatrixAggregator):
    """Aggregator for the Riemannian geometric mean."""

    def __init__(self, matrices: List[np.ndarray], weights: Optional[np.ndarray] = None, k_init: int = 0,
                 max_iter: int = 200, tolerance: float = 1e-12, corr_factor: float = 0):
        super().__init__(matrices, weights)
        self.k_init = k_init
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.corr_factor = corr_factor
        self.convergence_history: List[float] = []

    def _compute_method_specific_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the weights and the correlation matrix with riem_weights."""
        weights, corr_matrix = riem_weights(self.matrices)
        return weights, corr_matrix  # corr_matrix is our "RV matrix"

    def _geommean_two(self, A: np.ndarray, B: np.ndarray, t: float) -> np.ndarray:
        """Compute the weighted geometric mean between two matrices (A and B),
        with weight t (between 0 and 1)."""
        # Check if the matrices are positive definite: if not 'perturb' them
        if not (np.all(np.linalg.eigvalsh(A) > 0) and np.all(np.linalg.eigvalsh(B) > 0)):
            A = eigenval_perturb(A)
            B = eigenval_perturb(B)

        # Pick the best-conditioned matrix: exchange the matrices (if necessary),
        # to have the best-conditioned matrix as the first one.
        if np.linalg.cond(A) >= np.linalg.cond(B):
            A, B = B, A
            t = 1 - t

        # Compute the geometric mean
        lowRA = np.linalg.cholesky(A)
        uppRA = lowRA.T
        invchol1 = np.linalg.inv(lowRA)
        invchol2 = np.linalg.inv(uppRA)

        V = invchol1 @ B @ invchol2
        U, diag_vec, _ = np.linalg.svd(V, hermitian=True)
        D = np.diag(diag_vec)

        if self.corr_factor != 0:
            D = D + np.min(np.diag(D)) + self.corr_factor

        Dpower = np.diag(np.power(np.diag(D), t))
        middle = U @ Dpower @ U.T
        return lowRA @ middle @ uppRA

    def aggregate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.convergence_history = [] # Reset after every call
        if len(self.matrices) == 1:
            return self.matrices[0], {"method": "geometric_mean", "iterations": 0, "weights_source": self.weights_source,"RV_matrix": self.RV_matrix}

        if len(self.matrices) == 2: # Closed form for the weighted geometric mean
            result = self._geommean_two(self.matrices[1], self.matrices[0], self.weights[0])
            info = {"method": "geometric_mean", "iterations": 1, "weights_source": self.weights_source,"RV_matrix": self.RV_matrix}
            return result, info

        # Iterative algorithm for more than two matrices
        k = self.k_init  # Parameterizable initial index
        jk = modif_mod(k, len(self.matrices))  # Circular index
        X_current = self.matrices[jk - 1]  # Initial matrix

        for iter_count in range(1, self.max_iter + 1):
            jk_next = modif_mod(k + 1, len(self.matrices))
            w_exp = self.weights[jk_next - 1]
            S_next = self.matrices[jk_next - 1]

            # Denominator calculation
            denom = 0
            indices = np.array(list(range(k + 1))) + 1
            for i in indices:
                denom += self.weights[modif_mod(i, len(self.matrices)) - 1]

            t = w_exp / denom
            X_next = self._geommean_two(X_current, S_next, t)

            # Compute convergence error
            diff = X_next - X_current
            diff = X_next - X_current
            num_err = np.trace(diff @ diff.T)
            den_err = np.trace(X_current @ X_current.T)
            error = num_err / den_err
            self.convergence_history.append(error)

            # Check convergence
            if error <= self.tolerance:
                print(f"Convergence reached at iteration: {iter_count}")
                info = {
                    "method": "geometric_mean",
                    "iterations": iter_count,
                    "weights_source": self.weights_source,
                    "RV_matrix": self.RV_matrix,
                    "convergence_history": self.convergence_history.copy(),
                    "final_error": error
               }
                return X_next, info # In case of convergence, return X_next

            # Prepare next iteration
            X_current = X_next
            k += 1

        print('Maximum number of iterations reached')
        info = {
            "method": "geometric_mean",
            "iterations":  self.max_iter,
            "weights_source": self.weights_source,
            "RV_matrix": self.RV_matrix,
            "convergence_history": self.convergence_history.copy(),
            "final_error": self.convergence_history[-1] if self.convergence_history else float('inf') # Check if the list is empty for robustness
        }

        return X_current, info

class WassersteinAggregator(SimilarityMatrixAggregator):
    """Aggregator for the Wasserstein mean."""

    def __init__(self, matrices: List[np.ndarray], weights: Optional[np.ndarray] = None,
                 max_iter: int =  10, tolerance: float = 2e-8):
        super().__init__(matrices, weights)
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.convergence_history: List[float] = []

    def _compute_method_specific_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the weights and the correlation matrix with riem_weights."""
        weights, corr_matrix = riem_weights(self.matrices)
        return weights, corr_matrix

    # Helper function for the computation of the Wasserstein barycenter
    # when we have more than two matrices (fixed-point iteration).
    def _kx_compute(self, X: np.ndarray) -> np.ndarray:
        """Compute the operator K(X) for the Wasserstein algorithm, with X=X_current."""
        rad = square_root_matrix(X)
        negrad = np.linalg.inv(rad)
        to_sum = [np.zeros_like(X) for _ in range(len(self.matrices))]

        for i, matrix in enumerate(self.matrices):
            a = rad @ matrix @ rad
            sqa = square_root_matrix(a)
            to_sum[i] = self.weights[i] * sqa

        amount = np.sum(to_sum, axis=0)
        sq_amount = amount @ amount
        return negrad @ sq_amount @ negrad

    def aggregate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.convergence_history = [] # Reset of the convergence history
        if len(self.matrices) == 1:
            return self.matrices[0], {"method": "wasserstein_mean", "iterations": 0, "weights_source": self.weights_source,
            "RV_matrix": self.RV_matrix}

        if len(self.matrices) == 2:
            # Implementation of the closed form (for two matrices)
            s1 = (self.weights[0]**2) * self.matrices[0]
            s2 = (self.weights[1]**2) * self.matrices[1]
            s12 = self.weights[0] * self.weights[1] * (
                square_root_matrix(self.matrices[0] @ self.matrices[1]) +
                square_root_matrix(self.matrices[1] @ self.matrices[0]))
            result = s1 + s2 + s12
            info = {"method": "wasserstein_mean", "iterations": 1, "weights_source": self.weights_source,
            "RV_matrix": self.RV_matrix}
            return result, info

        # Iterative algorithm for more than two matrices
        import random
        k = random.randint(0, len(self.matrices) - 1) # Pick at random an initial matrix
        X_current = self.matrices[k]
        self.convergence_history = []

        for iter_count in range(1, self.max_iter + 1):
            X_next = self._kx_compute(X_current)

            # Compute convergence error
            diff = X_next - X_current
            error = np.trace(diff @ diff.T) / np.trace(X_current @ X_current.T)
            self.convergence_history.append(error)

            if error <= self.tolerance:
                print(f"Convergence reached at iteration: {iter_count}")
                info = {
                    "method": "wasserstein_mean",
                    "iterations": iter_count,
                    "weights_source": self.weights_source,
                    "RV_matrix": self.RV_matrix,
                    "convergence_history": self.convergence_history.copy(),
                    "final_error": error
                }
                return X_next, info

            X_current = X_next

        print('Maximum number of iterations reached')
        info = {
            "method": "wasserstein_mean",
            "iterations": self.max_iter,
            "weights_source": self.weights_source,
            "RV_matrix": self.RV_matrix,
            "convergence_history": self.convergence_history.copy(),
            "final_error": self.convergence_history[-1] if self.convergence_history else float('inf')
        }
        return X_current, info


class SNFAggregator(SimilarityMatrixAggregator):
    """Aggregator which uses Similarity Network Fusion (SNF) to integrate
    multiple similarity matrices."""

    def __init__(self, matrices: List[np.ndarray], weights: Optional[np.ndarray] = None,
                 K: int = 20, t: int = 20, alpha: float = 0.5):
        """Args:
            matrices: List of matrices we want to merge
            K: Number of nearest neighbors for the creation of the affinity matrix (default: 20)
            t: Number of fusion iterations (default: 20)
            alpha: Regularization parameter (default: 0.5)"""
        super().__init__(matrices, weights)
        self.K = K
        self.t = t
        self.alpha = alpha

    # To maintain the attributes corresponding to the weights (necessary for the three previous aggregators),
    # we set, for SNF (which doesn't need weights), uniform weights.
    def _compute_method_specific_weights(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """SNF doesn't use traditional weights, therefore we return uniform weights.
        The fusion occurs through the SNF algorithm itself."""
        return np.ones(len(self.matrices)) / len(self.matrices), None # La matrice RV è None

    def _affinity_matrix(self, W: np.ndarray, K: int) -> np.ndarray:
        """Compute the affinity matrix from a similarity matrix."""
        n = W.shape[0]
        affinity = np.zeros((n, n))

        for i in range(n):
            # Find the K nearest neighbors (excluding itself)
            indices = np.argsort(W[i])[::-1][1:K+1] # Indices of the most similar K
            for j in indices:
                affinity[i, j] = W[i, j]

        # Make it symmetric
        affinity = (affinity + affinity.T) / 2
        return affinity # Return a sparse matrix representing the local affinities
    def _normalized_cut(self, W: np.ndarray) -> np.ndarray:
        """Apply normalized cut to similarity matrix W.
        The goal is to normalize the matrix (row-wise normalization)
        to balance the influence of nodes with many connections."""
        # Compute the degree matrix (diagonal with sum of the rows of W)
        D = np.diag(np.sum(W, axis=1))

        # Compute D^(-1/2)
        D_sqrt_inv = np.linalg.pinv(np.sqrt(D)) # Pseudoinverse matrix

        # Normalized cut: D^(-1/2) * W * D^(-1/2)
        W_normalized = D_sqrt_inv @ W @ D_sqrt_inv
        return W_normalized

    def aggregate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply the SNF algorithm to merge the similarity matrices."""
        if len(self.matrices) == 1:
            info = {
                "method": "snf",
                "iterations": 0,
                "RV_matrix": self.RV_matrix,  # To maintain consistency
                "weights_source": self.weights_source,
                "parameters": {"K": self.K, "t": self.t, "alpha": self.alpha}
            }
            return self.matrices[0], info

        # Normalize every similarity matrix
        W_normalized = []
        for W in self.matrices: # W is a similarity matrix
            W_norm = self._normalized_cut(W)
            W_normalized.append(W_norm)

        # Create local affinity matrices for every view
        S_local = []
        for W in W_normalized:
            S = self._affinity_matrix(W, self.K)
            S_local.append(S)

        # Iterative fusion: W_current is a list of matrices,
        # each one of them represents a view and, in each iteration, every view is updated according to the others
        W_current = W_normalized.copy()

        for iteration in range(self.t): # Fusion iteration
            W_new = []
            for i in range(len(W_current)):
                # Combine similarities from other views (every matrix 
                # in self.matrices represents a different 'view' on the data)
                other_views = [W_current[j] for j in range(len(W_current)) if j != i]

                if other_views:
                    # Mean of the other views
                    W_other = np.mean(other_views, axis=0)
                else:
                    W_other = W_current[i]  # If there's only one matrix, it uses itself

                # Update the similarity: fusion of the current view with the others
                W_update = S_local[i] @ W_other @ S_local[i].T # Where S_local[i] is the local affinity matrix of the view i

                # Apply the normalized cut to W_update and regularize
                W_update = self._normalized_cut(W_update)

                # Regularization
                W_update = self.alpha * W_update + (1 - self.alpha) * W_current[i]
                W_new.append(W_update)

            W_current = W_new

        # Final fused matrix (mean of all the views): mean of all the affinity matrices
        # after t-iterations
        W_fused = np.mean(W_current, axis=0)

        info = {
            "method": "snf",
            "iterations": self.t,
            "RV_matrix": self.RV_matrix,
            "weights_source": self.weights_source,
            "parameters": {"K": self.K, "t": self.t, "alpha": self.alpha}
        }

        return W_fused, info

# Useful function for aggregation
def aggregate(matrices: List[np.ndarray], method: str = 'snf', weights: Optional[np.ndarray] = None, **kwargs) ->  Dict[str, Any]:
    """Helper function for the aggregation of a list of matrices using different methods.
    Args:
        matrices: List of similarity matrices
        method: Aggregation method ('weighted_mean', 'geometric', 'wasserstein', 'snf')
        weights: Optional weights
        **kwargs: Additional method-specific parameters
    Returns:
        Aggregated matrix and some info about the process"""
    # Define a dictionary which associates every string 'method' to the corresponding class
    aggregators = {
        'weighted_mean': WeightedMeanAggregator,
        'geometric': GeometricAggregator,
        'wasserstein': WassersteinAggregator,
        'snf': SNFAggregator
    }

    if method not in aggregators:
        raise ValueError(f"Method not supported: {method}. Methods available: {list(aggregators.keys())}")
    aggregator = aggregators[method](matrices, weights=weights, **kwargs)
    result, info = aggregator.aggregate()
    normalized_result = normalize_simmat(result)
    # Return the completed dictionary (with the original version of the barycenter, the normalized version, 
    # so that the diagonal entries are equal to 1, and info about the aggregation method used)
    return {
        'aggregated_matrix': result,
        'normalized_aggregated_matrix': normalized_result,
        'info': info
    }
