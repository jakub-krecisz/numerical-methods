import matplotlib.pyplot as plt
import numpy as np
import time


def get_solution_by_numpy_lib(size):
    a_matrix = np.diag([0.2] * (size - 1), -1)
    a_matrix += np.diag([1.2] * size)
    a_matrix += np.diag([0.1 / i for i in range(1, size)], 1)
    a_matrix += np.diag([0.4 / i ** 2 for i in range(1, size - 1)], 2)
    x_vector = np.array([_ for _ in range(1, size + 1)])

    start_time = time.time()
    solution = np.linalg.solve(a_matrix, x_vector)

    return solution, time.time() - start_time


def get_solution_numerically(size):
    a_matrix = [[0] + [0.2] * (size - 1), [1.2] * size, [0.1 / i for i in range(1, size)] + [0],
                [0.4 / i ** 2 for i in range(1, size - 1)] + [0] + [0]]

    x_vector = [_ for _ in range(1, size + 1)]

    start_time = time.time()
    for i in range(1, size - 2):
        a_matrix[0][i] = a_matrix[0][i] / a_matrix[1][i - 1]
        a_matrix[1][i] = a_matrix[1][i] - a_matrix[0][i] * a_matrix[2][i - 1]
        a_matrix[2][i] = a_matrix[2][i] - a_matrix[0][i] * a_matrix[3][i - 1]

    a_matrix[0][size - 2] = a_matrix[0][size - 2] / a_matrix[1][size - 3]
    a_matrix[1][size - 2] = a_matrix[1][size - 2] - a_matrix[0][size - 2] * a_matrix[2][size - 3]
    a_matrix[2][size - 2] = a_matrix[2][size - 2] - a_matrix[0][size - 2] * a_matrix[3][size - 3]

    a_matrix[0][size - 1] = a_matrix[0][size - 1] / a_matrix[1][size - 2]
    a_matrix[1][size - 1] = a_matrix[1][size - 1] - a_matrix[0][size - 1] * a_matrix[2][size - 2]

    # Forward Substitution
    for i in range(1, size):
        x_vector[i] = x_vector[i] - a_matrix[0][i] * x_vector[i - 1]

    # Backward Substitution
    x_vector[size - 1] = x_vector[size - 1] / a_matrix[1][size - 1]
    x_vector[size - 2] = (x_vector[size - 2] - a_matrix[2][size - 2] * x_vector[size - 1]) / a_matrix[1][size - 2]
    for i in range(size - 3, -1, -1):
        x_vector[i] = (x_vector[i] - a_matrix[3][i] * x_vector[i + 2] - a_matrix[2][i] * x_vector[i + 1])\
                      / a_matrix[1][i]

    # Determinant
    det_a = 1
    for val in a_matrix[1]:
        det_a *= val

    return x_vector, det_a, time.time() - start_time


def generate_graph():
    numpy_results, numerical_results = {}, {}

    for size in range(100, 6000, 200):
        numpy_results[size] = get_solution_by_numpy_lib(size)[1] * 1000000
        numerical_results[size] = get_solution_numerically(size)[2] * 1000000

    plt.grid(True)
    plt.title('Computing time')
    plt.xlabel('Matrix dimension (N)')
    plt.ylabel('Microseconds (μs)')
    plt.loglog(numpy_results.keys(), numpy_results.values(), 'tab:green')
    plt.loglog(numerical_results.keys(), numerical_results.values(), 'tab:red')
    plt.loglog(numerical_results.keys(), np.array(list(numerical_results.keys())), 'tab:gray')
    plt.loglog(numpy_results.keys(), np.array(list(numpy_results.keys())) ** 2, 'tab:gray')
    plt.legend(['Solving time by numPy library', 'Solving time numerically', 'f(x) = x', 'f(x) = x^2'])
    plt.show()


if __name__ == '__main__':
    n = 100
    print(f'The solution of the equation for N = {n}:\n{get_solution_numerically(n)[0]}\n'
          f'The determinant of A is equal: det(A) = {get_solution_numerically(n)[1]}')
    generate_graph()
