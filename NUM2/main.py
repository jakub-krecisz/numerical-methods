import numpy as np

from config import A1_MATRIX, A2_MATRIX, B_VECTOR, B_VECTOR_PRIM


def solve_matrix(matrix_a: np.ndarray[float], matrix_b: np.ndarray[float]) -> np.ndarray[float]:
    """
    Solves a linear system of equations for the given matrix A and vector B.

    :param matrix_a: Coefficient matrix A.
    :param matrix_b: Right-hand side vector B.
    :return: Solution vector y.
    """
    return np.linalg.solve(matrix_a, matrix_b)


def get_norm(matrix: np.ndarray[float]) -> float:
    """
    Computes the norm of the given matrix.

    :param matrix: Input matrix.
    :return: Norm of the matrix.
    """
    return np.linalg.norm(matrix)


def get_condition(matrix: np.ndarray[float]) -> float:
    """
    Computes the condition number of the given matrix.

    :param matrix: Input matrix.
    :return: Condition number of the matrix.
    """
    return np.linalg.cond(matrix)


if __name__ == '__main__':
    y1 = solve_matrix(A1_MATRIX, B_VECTOR)
    y2 = solve_matrix(A2_MATRIX, B_VECTOR)
    y1_prim = solve_matrix(A1_MATRIX, B_VECTOR_PRIM)
    y2_prim = solve_matrix(A2_MATRIX, B_VECTOR_PRIM)

    print(f"A₁y₁ = B:\n\ty₁ = {y1}\n")
    print(f"A₁y₁' = B':\n\ty₁' = {y1_prim}\n")
    print(f"A₂y₂ = B:\n\ty₂ = {y2}\n")
    print(f"A₂y₂' = B':\n\ty₂' = {y2_prim}\n")

    delta1 = get_norm(y1 - y1_prim)
    delta2 = get_norm(y2 - y2_prim)
    condition1 = get_condition(A1_MATRIX)
    condition2 = get_condition(A2_MATRIX)

    print(f"Δ₁ = ||y₁ - y₁'||₂\n\t Δ₁ = {delta1:.20f}\n")
    print(f"Δ₂ = ||y₂ - y₂'||₂\n\t Δ₂ = {delta2:.20f}\n")
    print(f"cond(A₁) = {condition1:.20f}\n")
    print(f"cond(A₂) = {condition2:.20f}\n")
