import numpy as np
import matplotlib.pyplot as plt

from config import N, B_VECTOR, X_VECTOR, MAX_NUM_OF_ITERATIONS, PRECISION


def solve_by_jacobian_method(xVector, max_iterations, size, bVector):
    iterativeApprox = []
    previousNorm = 0
    for _ in range(max_iterations):
        copyOfxVector = xVector.copy()
        for index in range(size):
            if index == 0:
                xVector[index] = (bVector[index] - xVector[index + 1] - 0.2 * xVector[index + 2]) / 3
            elif index == 1:
                xVector[index] = (bVector[index] - copyOfxVector[index - 1] - xVector[index + 1] - 0.2 * xVector[
                    index + 2]) / 3
            elif index == size - 2:
                xVector[index] = (bVector[index] - copyOfxVector[index - 1] - 0.2 * copyOfxVector[index - 2] - xVector[
                    index + 1]) / 3
            elif index == size - 1:
                xVector[index] = (bVector[index] - copyOfxVector[index - 1] - 0.2 * copyOfxVector[index - 2]) / 3
            else:
                xVector[index] = (bVector[index] - copyOfxVector[index - 1] - 0.2 * copyOfxVector[index - 2] - xVector[
                    index + 1] - 0.2 * xVector[index + 2]) / 3
        actualNorm = np.sqrt(sum(map(lambda a, b: (a - b) ** 2, xVector, copyOfxVector)))

        iterativeApprox.append(xVector.copy())

        # Sprawdzenie czy zbiegło
        if abs(previousNorm - actualNorm) < 10 ** (-12):
            break
        previousNorm = actualNorm

    return iterativeApprox.pop(), xVector


def solve_by_gauss_seidel_method(xVector, max_iterations, size, bVector):
    iterativeApprox = []
    previousNorm = 0
    for _ in range(max_iterations):
        copyOfxVector = xVector.copy()
        for index in range(size):
            if index == 0:
                xVector[index] = (bVector[index] - xVector[index + 1] - 0.2 * xVector[index + 2]) / 3
            elif index == 1:
                xVector[index] = (bVector[index] - xVector[index - 1] - xVector[index + 1] - 0.2 * xVector[
                    index + 2]) / 3
            elif index == size - 2:
                xVector[index] = (bVector[index] - xVector[index - 1] - 0.2 * xVector[index - 2] - xVector[
                    index + 1]) / 3
            elif index == size - 1:
                xVector[index] = (bVector[index] - xVector[index - 1] - 0.2 * xVector[index - 2]) / 3
            else:
                xVector[index] = (bVector[index] - xVector[index - 1] - 0.2 * xVector[index - 2] - xVector[
                    index + 1] - 0.2 * xVector[index + 2]) / 3

        actualNorm = np.sqrt(sum(map(lambda a, b: (a - b) ** 2, xVector, copyOfxVector)))

        iterativeApprox.append(xVector.copy())

        # Sprawdzenie czy zbiegło
        if abs(previousNorm - actualNorm) < 10 ** PRECISION:
            break

        previousNorm = actualNorm

    return iterativeApprox.pop(), xVector


def generate_graph(firstDiffs, secondDiffs, title):
    plt.grid(True)
    plt.xlabel('n')
    plt.ylabel("$|x(n) - x[-1]|$")
    plt.yscale('log')
    plt.plot(list(range(1, len(firstDiffs) + 1)), firstDiffs, 'tab:green')
    plt.plot(list(range(1, len(secondDiffs) + 1)), secondDiffs, 'tab:red')
    plt.legend(['Jacobian Method', 'Gauss-Seidel method'])
    plt.title(f'Comparison iterative methods; x with all {title} values')
    plt.show()


if __name__ == '__main__':
    for key, xValues in X_VECTOR.items():
        iterativeApproxJacobi, solutionJacobi = solve_by_jacobian_method(xValues.copy(),
                                                                         MAX_NUM_OF_ITERATIONS, N,
                                                                         B_VECTOR)
        iterativeApproxGaussSeidel, solutionGaussSeidel = solve_by_gauss_seidel_method(xValues.copy(),
                                                                                       MAX_NUM_OF_ITERATIONS, N,
                                                                                       B_VECTOR)

        print(f'Solution by Jacobi method:\n{solutionJacobi}')
        print(f'Solution by Gauss-Seidel method:\n{solutionGaussSeidel}')

        # Obliczanie różnicy pomiędzy rozwiązaniem w poszczególnych iteracjach a rozwiązaniem dokładnym
        diffsJacobi = []
        for approximateSolution in iterativeApproxJacobi:
            diffsJacobi.append(np.sqrt(sum(map(lambda a, b: (a - b) ** 2, approximateSolution, solutionJacobi))))

        diffsGaussSeidel = []
        for approximateSolution in iterativeApproxGaussSeidel:
            diffsGaussSeidel.append(
                np.sqrt(sum(map(lambda a, b: (a - b) ** 2, approximateSolution, solutionGaussSeidel))))

        generate_graph(diffsJacobi, diffsGaussSeidel, title=key)
