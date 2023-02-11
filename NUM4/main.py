import time
import numpy as np
import matplotlib.pyplot as plt


def get_solution_by_numpy_lib(size):
    a_matrix = np.ones((size, size))
    a_matrix += np.diag([9] * size)
    a_matrix += np.diag([7] * (size - 1), 1)
    b_matrix = np.array([5] * size).transpose()

    start_time = time.time()
    result = np.linalg.solve(a_matrix, b_matrix)
    return time.time() - start_time, result


def get_solution_by_sherman_morrison(size):
    band_matrix = [[9] * size, [7] * (size - 1) + [0]]
    b_matrix = [5] * size
    z_vector, x_vector = [0] * size, [0] * size
    result_vector = []

    start_time = time.time()
    z_vector[-1] = b_matrix[size - 1] / band_matrix[0][size - 1]
    x_vector[-1] = 1 / band_matrix[0][size - 1]

    for index in range(size - 2, -1, -1):
        z_vector[index] = (b_matrix[-2] - band_matrix[1][index] * z_vector[index + 1]) / band_matrix[0][index]
        x_vector[index] = (1 - band_matrix[1][index] * x_vector[index + 1]) / band_matrix[0][index]

    delta = sum(z_vector) / (1 + sum(x_vector))

    for z_val, x_val in zip(z_vector, x_vector):
        result_vector.append(z_val - x_val * delta)

    return time.time() - start_time, result_vector


def generate_graph(do: str):
    numpy_results = {}
    algorythm_results = {}
    for size in range(100, 10000, 200):
        numpy_results[size] = get_solution_by_numpy_lib(size)[0] * 1000000
        algorythm_results[size] = get_solution_by_sherman_morrison(size)[0] * 1000000

    plt.grid(True)
    plt.title('Solving time')
    plt.xlabel('Matrix dimension (N)')
    plt.ylabel('Microseconds (μs)')
    plt.loglog(numpy_results.keys(), numpy_results.values(), 'tab:green')
    plt.loglog(algorythm_results.keys(), algorythm_results.values(), 'tab:red')
    plt.loglog(numpy_results.keys(), np.array(list(numpy_results.keys())), 'tab:gray')
    plt.loglog(numpy_results.keys(), np.array(list(numpy_results.keys()))**2, 'tab:gray')

    plt.legend(['Solving time by numPy library', 'Solving time by algorythm', 'F(x) = x', 'F(x) = x^2'])
    plt.show() if do == 'show' else plt.savefig('solving_time.svg')


if __name__ == '__main__':
    print(f'The solution of the equation Ay = b for N = 50:\n{get_solution_by_sherman_morrison(50)[1]}')
    generate_graph('show')
