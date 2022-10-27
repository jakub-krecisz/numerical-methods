import numpy as np

from config import a1, a2, b, bPrim


def solveMatrix(a, b):
    return np.linalg.solve(a, b)


def getNorm(matrix):
    return np.linalg.norm(matrix)


def getCondition(matrix):
    return np.linalg.cond(matrix)


if __name__ == '__main__':
    y1 = solveMatrix(a1, b)
    y2 = solveMatrix(a2, b)
    y1Prim = solveMatrix(a1, bPrim)
    y2Prim = solveMatrix(a2, bPrim)

    print(f"Solution for A\u2081y\u2081 = B:\n\ty\u2081 = {y1}\n")
    print(f"Solution for A\u2081y\u2081' = B':\n\ty\u2081' = {y1Prim}\n")
    print(f"Solution for A\u2082y\u2082 = B:\n\ty\u2082 = {y2}\n")
    print(f"Solution for A\u2082y\u2082' = B':\n\ty\u2082' = {y2Prim}\n")

    print("Δ\u2081 = ||y\u2081 - y\u2081'||\u2082\n\t Δ\u2081 = {:.20f}\n".format(getNorm(y1 - y1Prim)))
    print("Δ\u2082 = ||y\u2082 - y\u2082'||\u2082\n\t Δ\u2082 = {:.20f}\n".format(getNorm(y2 - y2Prim)))

    print("cond(A\u2081) = {:.20f}\n".format(getCondition(a1)))
    print("cond(A\u2082) = {:.20f}\n".format(getCondition(a2)))
