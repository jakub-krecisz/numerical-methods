import numpy as np
import matplotlib.pyplot as plt

from config import N, B_VECTOR, X_VECTOR, MAX_NUM_OF_ITERATIONS, PRECISION


def solve_by_jacobin_method(xVector, max_iterations, size, bVector):
    iterativeApprox = []
    previousNorm = 0
    for _ in range(max_iterations):
        copyOfxVector = xVector.copy()
        for index in range(size):
            if index == 0:
                xVector[index] = (bVector[index] - xVector[index + 1] - 0.2 * xVector[index + 2]) / 3
            elif index == 1:
                xVector[index] = (bVector[index] - copyOfxVector[index - 1] - xVector[index + 1] - 0.2 *
                                  xVector[index + 2]) / 3
            elif index == size - 2:
                xVector[index] = (bVector[index] - copyOfxVector[index - 1] - 0.2 * copyOfxVector[index - 2] -
                                  xVector[index + 1]) / 3
            elif index == size - 1:
                xVector[index] = (bVector[index] - copyOfxVector[index - 1] - 0.2 * copyOfxVector[index - 2]) / 3
            else:
                xVector[index] = (bVector[index] - copyOfxVector[index - 1] - 0.2 * copyOfxVector[index - 2] -
                                  xVector[index + 1] - 0.2 * xVector[index + 2]) / 3

        actualNorm = np.sqrt(sum(map(lambda a, b: (a - b) ** 2, xVector, copyOfxVector)))

        iterativeApprox.append(xVector.copy())

        # convergence check
        if abs(previousNorm - actualNorm) < 10 ** PRECISION:
            break
        else:
            previousNorm = actualNorm

    return iterativeApprox


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
                xVector[index] = (bVector[index] - xVector[index - 1] - 0.2 * xVector[index - 2] -
                                  xVector[index + 1] - 0.2 * xVector[index + 2]) / 3

        # checking the norm for actual iteration
        actualNorm = np.sqrt(sum(map(lambda a, b: (a - b) ** 2, xVector, copyOfxVector)))

        iterativeApprox.append(xVector.copy())

        # convergence check
        if abs(previousNorm - actualNorm) < 10 ** PRECISION:
            break
        else:
            previousNorm = actualNorm

    return iterativeApprox


def generate_graph(firstDiffs, secondDiffs, titleStr):
    plt.grid(True)
    plt.xlabel('Number of iterations (n)')
    plt.ylabel("$|x(n) - x[-1]|$")
    plt.yscale('log')
    plt.plot(list(range(1, len(firstDiffs) + 1)), firstDiffs, 'tab:green')
    plt.plot(list(range(1, len(secondDiffs) + 1)), secondDiffs, 'tab:red')
    plt.legend(['Jacobin Method', 'Gauss-Seidel method'])
    plt.title(f'Comparison iterative methods; x with all {titleStr} values')
    plt.show()


if __name__ == '__main__':
    for title, xVect in X_VECTOR.items():
        iterativeApproxJacobi = solve_by_jacobin_method(xVect.copy(), MAX_NUM_OF_ITERATIONS, N, B_VECTOR)
        iterativeApproxGaussSeidel = solve_by_gauss_seidel_method(xVect.copy(), MAX_NUM_OF_ITERATIONS, N, B_VECTOR)

        # Last iteration is our solution in that case
        print(f'Solution by Jacobi method:\n{iterativeApproxJacobi[-1]}')
        print(f'Solution by Gauss-Seidel method:\n{iterativeApproxGaussSeidel[-1]}')

        # Calculate the difference between all iteration and the last iteration
        diffsJacobi, diffsGaussSeidel = [], []
        for approximateSolution in iterativeApproxJacobi:
            diffsJacobi.append(
                np.sqrt(sum(map(lambda a, b: (a - b) ** 2, approximateSolution, iterativeApproxJacobi[-1]))))
        for approximateSolution in iterativeApproxGaussSeidel:
            diffsGaussSeidel.append(
                np.sqrt(sum(map(lambda a, b: (a - b) ** 2, approximateSolution, iterativeApproxGaussSeidel[-1]))))

        # Generate the graph
        generate_graph(diffsJacobi, diffsGaussSeidel, titleStr=title)
