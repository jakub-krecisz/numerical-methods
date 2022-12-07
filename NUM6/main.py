import numpy as np

from tabulate import tabulate
from config import PRECISION, MAX_ITERATIONS, M_MATRIX, B_MATRIX, Y_VECTOR

def _iterate_qr(matrix: np.array) -> np.array:
    """
    Function returns a single iteration of QR algorithm
    :param matrix: matrix which we are iterating by QR algorithm
    :return: matrix after one iteration of QR algorithm
    """
    matrixQ, matrixR = np.linalg.qr(matrix)
    return np.matmul(matrixR, matrixQ)


def _iterate_power(matrix, vectorY):
    """
    Function returns a single iteration of power method
    :param matrix: matrix for which we calculate the eigenvalue
    :param vectorY: previous value of vector Y
    :return: return the next iteration of vector Y
    """
    vectorZ = np.matmul(matrix, vectorY)
    return vectorZ / np.linalg.norm(vectorZ)


def get_eigenvalues_by_qr_algorithm(matrix: np.array) -> list:
    """
    Function determines all eigenvalues for given matrix by QR algorithm.
    :param matrix: matrix from which we want to determine the eigenvalues
    :return: list of eigenvalues
    """
    result = _iterate_qr(matrix)
    for _ in range(MAX_ITERATIONS):
        result = _iterate_qr(result)
        if all(abs(result[i][i - 1]) < PRECISION for i in range(1, len(result))):
            break
    return [result[x][x] for x in range(len(result))][::-1]


def get_eigenvector_by_power_method(matrix, vectorY):
    """
    Function determines the eigenvector for the corresponding largest eigenvalue.
    :param matrix: matrix from which we want to determine the eigenvector
    :param vectorY: initial arbitrary vector
    :return: eigenvector for the corresponding largest eigenvalue
    """
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
