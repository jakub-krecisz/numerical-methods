import numpy as np

from tabulate import tabulate
from config import PRECISION, MAX_ITERATIONS, M_MATRIX, B_MATRIX, Y_VECTOR

def _iterate_qr(matrix: np.array) -> np.array:
    matrixQ, matrixR = np.linalg.qr(matrix)
    return np.matmul(matrixR, matrixQ)


def _iterate_power(matrix, vectorY):
    vectorZ = np.matmul(matrix, vectorY)
    return vectorZ / np.linalg.norm(vectorZ)


def get_eigenvalues_by_qr_algorithm(matrix: np.array) -> list:
    result = _iterate_qr(matrix)
    for _ in range(MAX_ITERATIONS):
        result = _iterate_qr(result)
        if all(abs(result[i][i - 1]) < PRECISION for i in range(1, len(result))):
            break
    return [result[x][x] for x in range(len(result))][::-1]


def get_eigenvector_by_power_method(matrix, vectorY):
    result = _iterate_power(matrix, vectorY)
    for _ in range(MAX_ITERATIONS):
        previousIteration = result
        result = _iterate_power(matrix, result)
        if np.linalg.norm(previousIteration - result) < PRECISION:
            break
    return result


if __name__ == '__main__':
    print(f'Eigenvalues by QR algorithm:\n{get_eigenvalues_by_qr_algorithm(M_MATRIX)}')
    print(f'Eigenvalues by numPy library:\n{np.linalg.eig(M_MATRIX)[0]}\n')

    eigenvectorPowerMethod = get_eigenvector_by_power_method(B_MATRIX, Y_VECTOR)
    print('Greatest eigenvalue calculated by power method for matrix B: '
          f'{np.matmul(B_MATRIX, eigenvectorPowerMethod)[0] / eigenvectorPowerMethod[0]}')
    print(f'Corresponding eigenvector: {eigenvectorPowerMethod}\n')
    print(f'Greatest eigenvalue calculated by numPy library for matrix B: {max(np.linalg.eig(B_MATRIX)[0])}')
    print(f'Corresponding eigenvectors: \n{tabulate(np.linalg.eigh(B_MATRIX)[1])}')
