import numpy as np

from config import A1_MATRIX, A2_MATRIX, B_VECTOR, B_VECTOR_PRIM


def solve_matrix(a, b):
    return np.linalg.solve(a, b)


def get_norm(matrix):
    return np.linalg.norm(matrix)


def get_condition(matrix):
    return np.linalg.cond(matrix)


if __name__ == '__main__':
    y1 = solve_matrix(A1_MATRIX, B_VECTOR)
    y2 = solve_matrix(A2_MATRIX, B_VECTOR)
    y1_prim = solve_matrix(A1_MATRIX, B_VECTOR_PRIM)
    y2_prim = solve_matrix(A2_MATRIX, B_VECTOR_PRIM)

    print(f"Solution for A\u2081y\u2081 = B:\n\ty\u2081 = {y1}\n")
    print(f"Solution for A\u2081y\u2081' = B':\n\ty\u2081' = {y1_prim}\n")
    print(f"Solution for A\u2082y\u2082 = B:\n\ty\u2082 = {y2}\n")
    print(f"Solution for A\u2082y\u2082' = B':\n\ty\u2082' = {y2_prim}\n")

    print("Δ\u2081 = ||y\u2081 - y\u2081'||\u2082\n\t Δ\u2081 = {:.20f}\n".format(get_norm(y1 - y1_prim)))
    print("Δ\u2082 = ||y\u2082 - y\u2082'||\u2082\n\t Δ\u2082 = {:.20f}\n".format(get_norm(y2 - y2_prim)))

    print("cond(A\u2081) = {:.20f}\n".format(get_condition(A1_MATRIX)))
    print("cond(A\u2082) = {:.20f}\n".format(get_condition(A2_MATRIX)))
