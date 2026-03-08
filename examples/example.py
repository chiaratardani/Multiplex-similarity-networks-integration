import numpy as np
from multiplex_similarity_networks_integration.aggregation import aggregate

'''Now we generate, to show how to use the code, three similarity matrices
(completely positive matrices) at random, with dimensions 3 x 3.'''
# Parameters
n = 3       # Matrix dimensions (n x n)
k = 5       # Internal dimensions (number of columns of the matrix B)
num_matr = 3  # Number of matrices we want to generate

matrices = []
for _ in range(num_matr):
    # Generate a matrix B (n x k) at random, with non-negative entries.
    # np.random.rand, in fact, generates random elements from a uniform
    # distribution on [0,1].
    B = np.random.rand(n, k)  # Elements in [0, 1]
    
    # Compute the completely positive matrix A = B * B^T
    A = B @ B.T
    
    # Check if the entries are non-negative (they must be for construction):
    # the condition we want to test is np.all(A >= 0); therefore, we generate 
    # a boolean matrix (n x n), in which an element is True (1) if 
    # the corresponding entry in A is >= 0, otherwise it's False (0). The following string is the 
    # error message that must return in case the condition is not satisfied.
    assert np.all(A >= 0), "The matrix is not completely positive!" 
    matrices.append(A)

# Show the matrices generated at random
for i, A in enumerate(matrices):
    print(f"Matrice {i+1}:\n{A}\n")

'''Now we can use the function 'aggregate'
to obtain the mean of these matrices with the chosen
aggregation method (inserted as input).'''
snf_result = aggregate(matrices, 'snf')
frob_result = aggregate(matrices, 'weighted_mean')
riem_result = aggregate(matrices, 'geometric')
wass_result = aggregate(matrices, 'wasserstein')
