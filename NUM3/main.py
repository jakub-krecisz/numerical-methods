from functools import reduce
import numpy as np
import time


# Testy
def checkNumpy():
    A = np.diag([0.2] * (n - 1), -1)
    A += np.diag([1.2] * n)
    A += np.diag([0.1 / i for i in range(1, n)], 1)
    A += np.diag([0.4 / i ** 2 for i in range(1, n - 1)], 2)
    x = list(range(1, n + 1))

    start = time.time()

    np.linalg.solve(A, x)

    print("Czas numpy to: {:.20f}".format(time.time() - start))


n = 100

# Uzupełnienie diagonali macierzy A
matrix = []
matrix.append([0] + [0.2] * (n - 1))
matrix.append([1.2] * n)
matrix.append([0.1 / i for i in range(1, n)] + [0])
matrix.append([0.4 / i ** 2 for i in range(1, n - 1)] + [0] + [0])

# Stworzenie wektora wyrazów wolnych
x = list(range(1, n + 1))

start = time.time()

# Rozkład LU
for i in range(1, n - 2):
    matrix[0][i] = matrix[0][i] / matrix[1][i - 1]
    matrix[1][i] = matrix[1][i] - matrix[0][i] * matrix[2][i - 1]
    matrix[2][i] = matrix[2][i] - matrix[0][i] * matrix[3][i - 1]

matrix[0][n - 2] = matrix[0][n - 2] / matrix[1][n - 3]
matrix[1][n - 2] = matrix[1][n - 2] - matrix[0][n - 2] * matrix[2][n - 3]
matrix[2][n - 2] = matrix[2][n - 2] - matrix[0][n - 2] * matrix[3][n - 3]

matrix[0][n - 1] = matrix[0][n - 1] / matrix[1][n - 2]
matrix[1][n - 1] = matrix[1][n - 1] - matrix[0][n - 1] * matrix[2][n - 2]

# Podstawianie w przód
for i in range(1, n):
    x[i] = x[i] - matrix[0][i] * x[i - 1]

# Podstawiania w tył
x[n - 1] = x[n - 1] / matrix[1][n - 1]
x[n - 2] = (x[n - 2] - matrix[2][n - 2] * x[n - 1]) / matrix[1][n - 2]

for i in range(n - 3, -1, -1):
    x[i] = (x[i] - matrix[3][i] * x[i + 2] - matrix[2][i] * x[i + 1]) / matrix[1][i]

# Obliczanie wartości wyznacznika macierzy
wyznacznik = reduce(lambda a, b: a * b, matrix[1])
end = time.time() - start

print("Szukane rozwiązanie to: ", x)
print()
print("Wyznacznik macierzy A = ", wyznacznik)

# Testy
print("Czas programu to: {:.20f}".format(end))
checkNumpy()
