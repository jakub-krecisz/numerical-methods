from typing import List

import numpy as np
import matplotlib.pyplot as plt

from config import N, B_VECTOR, X_VECTOR, MAX_NUM_OF_ITERATIONS, PRECISION


def solve_by_jacobi_method(x_vector: List[float], b_vector: List[float],
                           size: int, max_iterations: int) -> List[List[float]]:
    """
    Solve the given equation using the Jacobi method to the specified precision or maximum number of iterations.

    :param x_vector: Initial vector x
    :param b_vector: Vector b from the equation
    :param size: Matrix dimension
    :param max_iterations: Maximum number of iterations
    :return: List of approximations for each iteration
    """
    iterative_approx = []
    previous_norm = 0
    for _ in range(max_iterations):
        copy_ofx_vector = x_vector.copy()
        for index in range(size):
            if index == 0:
                x_vector[index] = (b_vector[index] - x_vector[index + 1] - 0.2 * x_vector[index + 2]) / 3
            elif index == 1:
                x_vector[index] = (b_vector[index] - copy_ofx_vector[index - 1] - x_vector[index + 1] - 0.2 *
                                   x_vector[index + 2]) / 3
            elif index == size - 2:
                x_vector[index] = (b_vector[index] - copy_ofx_vector[index - 1] - 0.2 * copy_ofx_vector[index - 2] -
                                   x_vector[index + 1]) / 3
            elif index == size - 1:
                x_vector[index] = (b_vector[index] - copy_ofx_vector[index - 1] - 0.2 * copy_ofx_vector[index - 2]) / 3
            else:
                x_vector[index] = (b_vector[index] - copy_ofx_vector[index - 1] - 0.2 * copy_ofx_vector[index - 2] -
                                   x_vector[index + 1] - 0.2 * x_vector[index + 2]) / 3

        # checking the norm for actual iteration
        actual_norm = np.linalg.norm(np.array(x_vector) - np.array(copy_ofx_vector))

        # saving approximation of actual iteration
        iterative_approx.append(x_vector.copy())

        # convergence check
        if abs(previous_norm - actual_norm) > 10 ** PRECISION:
            previous_norm = actual_norm
        else:
            break

    return iterative_approx


def solve_by_gauss_seidel_method(x_vector: List[float], b_vector: List[float],
                                 size: int, max_iterations: int) -> List[List[float]]:
    """
    Solve the given equation using the Gauss-Seidel method to the specified precision or maximum number of iterations.

    :param x_vector: Initial vector x
    :param b_vector: Vector b from the equation
    :param size: Matrix dimension
    :param max_iterations: Maximum number of iterations
    :return: List of approximations for each iteration
    """
    iterative_approx = []
    previous_norm = 0
    for _ in range(max_iterations):
        copy_ofx_vector = x_vector.copy()
        for index in range(size):
            if index == 0:
                x_vector[index] = (b_vector[index] - x_vector[index + 1] - 0.2 * x_vector[index + 2]) / 3
            elif index == 1:
                x_vector[index] = (b_vector[index] - x_vector[index - 1] - x_vector[index + 1] - 0.2 *
                                   x_vector[index + 2]) / 3
            elif index == size - 2:
                x_vector[index] = (b_vector[index] - x_vector[index - 1] - 0.2 * x_vector[index - 2] -
                                   x_vector[index + 1]) / 3
            elif index == size - 1:
                x_vector[index] = (b_vector[index] - x_vector[index - 1] - 0.2 * x_vector[index - 2]) / 3
            else:
                x_vector[index] = (b_vector[index] - x_vector[index - 1] - 0.2 * x_vector[index - 2] -
                                   x_vector[index + 1] - 0.2 * x_vector[index + 2]) / 3

        # checking the norm for actual iteration
        actual_norm = np.linalg.norm(np.array(x_vector) - np.array(copy_ofx_vector))

        # saving approximation of actual iteration
        iterative_approx.append(x_vector.copy())

        # convergence check
        if abs(previous_norm - actual_norm) > 10 ** PRECISION:
            previous_norm = actual_norm
        else:
            break

    return iterative_approx


def generate_graph(first_diffs: List[float], second_diffs: List[float], title_str: str) -> None:
    """
    Generate a graph based on the differences and their iteration number.

    :param first_diffs: List of differences for the Jacobi method
    :param second_diffs: List of differences for the Gauss-Seidel method
    :param title_str: Title for the graph
    """
    plt.grid(True)
    plt.xlabel('Number of iterations (n)')
    plt.ylabel("$|x(n) - x[-1]|$")
    plt.yscale('log')
    plt.plot(list(range(1, len(first_diffs) + 1)), first_diffs, 'tab:green')
    plt.plot(list(range(1, len(second_diffs) + 1)), second_diffs, 'tab:red')
    plt.legend(['Jacobi Method', 'Gauss-Seidel Method'])
    plt.title(f'Comparison of Iterative Methods; x with all {title_str} values')
    plt.show()


if __name__ == '__main__':
    for title, x_vect in X_VECTOR.items():
        iterative_approx_jacobi = solve_by_jacobi_method(x_vect.copy(), B_VECTOR, N, MAX_NUM_OF_ITERATIONS)
        iterative_approx_gauss_seidel = solve_by_gauss_seidel_method(x_vect.copy(), B_VECTOR, N, MAX_NUM_OF_ITERATIONS)

        # The last iteration gives us the solution
        print(f'Solution of the equation for N={N}, x vector with all {title} values')
        print(f'Solution by Jacobi method:\n{iterative_approx_jacobi[-1]}')
        print(f'Solution by Gauss-Seidel method:\n{iterative_approx_gauss_seidel[-1]}\n')

        # Calculate the difference between each iteration and the last iteration
        diffs_jacobi, diffs_gauss_seidel = [], []
        for iteration in range(len(iterative_approx_jacobi) - 1):
            diffs_jacobi.append(np.sqrt(sum(
                map(lambda a, b: abs(a - b), iterative_approx_jacobi[iteration], iterative_approx_jacobi[-1]))))
        for iteration in range(len(iterative_approx_gauss_seidel) - 1):
            diffs_gauss_seidel.append(np.sqrt(sum(
                map(lambda a, b: abs(a - b), iterative_approx_gauss_seidel[iteration], iterative_approx_gauss_seidel[-1]))))

        # Generate a graph
        generate_graph(diffs_jacobi, diffs_gauss_seidel, title_str=title)
