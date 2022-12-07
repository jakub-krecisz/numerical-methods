import numpy as np

# Defined precision at which algorithm will finish
PRECISION = 10 ** -16

# Maximal number of iteration
MAX_ITERATIONS = 500

M_MATRIX = np.array([[3, 6, 6, 9],
                     [1, 4, 0, 9],
                     [0, 0.2, 6, 12],
                     [0, 0, 0.1, 6]])

B_MATRIX = np.array([[3, 4, 2, 4],
                     [4, 7, 1, -3],
                     [2, 1, 3, 2],
                     [4, -3, 2, 2]])

Y_VECTOR = np.array([1 for _ in range(len(B_MATRIX))])

