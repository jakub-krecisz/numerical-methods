import random

N = 100

B_VECTOR = [x for x in range(1, N + 1)]

X_VECTOR = {'RANDOM': random.sample(range(1, 500), N),
            'ZEROS': [0 for _ in range(N)],
            'TENS': [10 for _ in range(N)]}

MAX_NUM_OF_ITERATIONS = 500

PRECISION = -16
