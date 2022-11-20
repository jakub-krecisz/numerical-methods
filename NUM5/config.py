import random

N = 100

B_VECTOR = [x for x in range(1, N + 1)]

X_VECTOR = {'random': random.sample(range(1, 500), N),
            'zeros': [0 for _ in range(N)],
            'tens': [10 for _ in range(N)]}

MAX_NUM_OF_ITERATIONS = 500

PRECISION = -16
