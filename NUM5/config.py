import random

# Our matrix dimension
N = 100

# Our given b vector from equation
B_VECTOR = [x for x in range(1, N + 1)]

# Example x vectors
X_VECTOR = {'random': random.sample(range(1, 500), N),
            'zeros': [0 for _ in range(N)],
            'ones': [1 for _ in range(N)],
            'nines': [9 for _ in range(N)]}

# Max number of iteration at which algorithm will finish
MAX_NUM_OF_ITERATIONS = 500

# Defined precision at which algorithm will finish
PRECISION = -16
