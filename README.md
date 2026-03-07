# Integration of multiplex similarity networks
## 💻 Author 
**Chiara Tardani**
## 🚀 Goal
The goal of the implemented functions is to integrate multiplex similarity networks, passed as input, in one single similarity network (monoplex similarity network). The aggregation methods proposed were taken from <https://github.com/DedeBac/SimilarityMatrixAggregation> for the three barycenters (Similarity Matrix Average, SMA): 
```markdown
```python
class WeightedMeanAggregator(SimilarityMatrixAggregator) # Aggregator for the weighted arithmetic mean with Frobenius
class GeometricAggregator(SimilarityMatrixAggregator) # Aggregator for the Riemannian geometric mean
class WassersteinAggregator(SimilarityMatrixAggregator) # Aggregator for the Wasserstein mean
```
and from <https://github.com/maxconway/SNFtool/tree/master> for the code (in R) corresponding to the SNF (Similarity Network Fusion) tool, which has been here implemented in Python and, as well as the three previous methods, as a class:
```markdown
```python
class SNFAggregator(SimilarityMatrixAggregator) # Aggregator which uses Similarity Network Fusion (SNF) 
```
The goal is, in fact, to exploit Python's object-oriented paradigm, creating classes that can represent different methods for the aggregation of similarity matrices, with the idea of ​​turning the code into a usable Python library.
## 📚 Structure
The repository contains the following scripts .py:

- **aggregation.py** : which contains the classes for the four aggregation methods proposed;

- **utils.py** : which contains some functions required by the majority of the methods;

- **example.py** : which contains an application example of the code.
## ✨ Characteristics
The heart of the project is the use of an abstract class, `SimilarityMatrixAggregator`:
```markdown
```python
 class SimilarityMatrixAggregator(ABC):
    """Abstract base class for the aggregators of similarity matrices."""
    
    def __init__(self, matrices: List[np.ndarray], weights: Optional[np.ndarray] = None):
        self.matrices = matrices
        self.weights, self.weights_source, self.RV_matrix = self._resolve_weights(weights)
        self._validate_input()
    
    def _validate_input(self) -> None:
        """Validate the input (similarity matrices)."""
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
        In particular, it returns (weights, source, RV matrix)."""
        if weights is not None:
            # Validation if weights were given
            if len(weights) != len(self.matrices):
                raise ValueError(f"Number of weights ({len(weights)}) != Number of matrices ({len(self.matrices)})")
            if not np.all(weights >= 0):
                raise ValueError("All the weights must be non-negative")
            if np.sum(weights) == 0:
                raise ValueError("The sum of the weights cannot be zero")
            
            normalized_weights = weights / np.sum(weights)  # Normalize
            return normalized_weights, "provided", None  # If the weights were given, no RV matrix is returned
        else:
            # Compute method-specific weights
            computed_weights, RV_matrix = self._compute_method_specific_weights()
            return computed_weights, "computed", RV_matrix
            
    # We use the @abstractmethod decorator to highlight the methods (abstract methods)
    # which have to be implemented in the subclasses.
    @abstractmethod
    def _compute_method_specific_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        pass            
    
    @abstractmethod
    def aggregate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute the aggregated matrix."""
        pass
        
    # The following method returns the weights used in the aggregation:
    # it returns an array NumPy or None if the weights aren't available
    # (type Optional is used as a precaution).
    def get_weights(self) -> Optional[np.ndarray]:
        """Return the weights used in the aggregation."""
        return self.weights
```
from which the above-mentioned subclasses derive, which inherit the methods of `SimilarityMatrixAggregator(ABC)` and override abstract methods (marked with the `@abstractmethod` decorator) consistently. In this way, it's easy to add new aggregators (and, therefore, new subclasses):
```markdown
```python
class NewAggregator(SimilarityMatrixAggregator):
  def _compute_method_specific_weights(self) -> Tuple[np.ndarray, np.ndarray]:
     """Here we implement the specific calculation of
     the weights for the NewAggregator method."""
     return custom_weights_calculation(self.matrices)
  def aggregate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
     """Here we implement the aggregation process."""
     pass
```
