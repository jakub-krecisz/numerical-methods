import numpy as np
import matplotlib.pyplot as plt

from config import N, B_VECTOR, X_VECTOR, MAX_NUM_OF_ITERATIONS, PRECISION


def solve_by_jacobin_method(xVector: list, bVector: list, size: int, max_iterations: int) -> list:
    """
    The function determines the solution of our given equation to the
    specified precision of the results or to the maximum number of iterations by Jacobin method.

    :param xVector: our vector x with example values
    :param bVector: our b vector from the equation
    :param size: our matrix dimension
    :param max_iterations: maximum number of iterations
    :return: list of approximations for all iterations
    """
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

        # checking the norm for actual iteration
        actualNorm = np.sqrt(sum(map(lambda a, b: (a - b) ** 2, xVector, copyOfxVector)))

        # saving approximation of actual iteration
        iterativeApprox.append(xVector.copy())

        # convergence check
        if abs(previousNorm - actualNorm) > 10 ** PRECISION:
            previousNorm = actualNorm
        else:
            break

    return iterativeApprox


def solve_by_gauss_seidel_method(xVector: list, bVector: list, size: int, max_iterations: int) -> list:
    """
    The function determines the solution of our given equation to the
    specified precision of the results or to the maximum number of iterations by Gauss-Seidel method.

    :param xVector: our vector x with example values
    :param bVector: our b vector from the equation
    :param size: our matrix dimension
    :param max_iterations: maximum number of iterations
    :return: list of approximations for all iterations
    """
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

        # saving approximation of actual iteration
        iterativeApprox.append(xVector.copy())

        # convergence check
        if abs(previousNorm - actualNorm) > 10 ** PRECISION:
            previousNorm = actualNorm
        else:
            break

    return iterativeApprox


def generate_graph(firstDiffs: list, secondDiffs: list, titleStr: str):
    """
    The function generates a graph based on the differences and their iteration number.

    :param firstDiffs: first difference list
    :param secondDiffs: second difference list
    :param titleStr: title of our graph
    """
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
        iterativeApproxJacobi = solve_by_jacobin_method(xVect.copy(), B_VECTOR, N, MAX_NUM_OF_ITERATIONS)
        iterativeApproxGaussSeidel = solve_by_gauss_seidel_method(xVect.copy(), B_VECTOR, N, MAX_NUM_OF_ITERATIONS)

        # Last iteration is our solution in that case
        print(f'Solution by Jacobi method:\n{iterativeApproxJacobi[-1]}')
        print(f'Solution by Gauss-Seidel method:\n{iterativeApproxGaussSeidel[-1]}')

        # Calculate the difference between all iteration and the last iteration
        diffsJacobi, diffsGaussSeidel = [], []
        for iteration in range(len(iterativeApproxJacobi) - 1):
            diffsJacobi.append(np.sqrt(sum(
                map(lambda a, b: (a - b) ** 2, iterativeApproxJacobi[iteration], iterativeApproxJacobi[-1]))))
        for iteration in range(len(iterativeApproxGaussSeidel) - 1):
            diffsGaussSeidel.append(np.sqrt(sum(
                map(lambda a, b: (a - b) ** 2, iterativeApproxGaussSeidel[iteration], iterativeApproxGaussSeidel[-1]))))

        # Generate a graph
        generate_graph(diffsJacobi, diffsGaussSeidel, titleStr=title)
