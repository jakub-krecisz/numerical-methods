import time
import numpy as np
import matplotlib.pyplot as plt


def get_solution_by_numpy_lib(size):
    aMatrix = np.ones((size, size))
    aMatrix += np.diag([9] * size)
    aMatrix += np.diag([7] * (size - 1), 1)
    bMatrix = np.array([5] * size).transpose()

    startTimeTime = time.time()
    result = np.linalg.solve(aMatrix, bMatrix)
    return time.time() - startTimeTime, result


def get_solution_by_sherman_morrison(size):
    bandMatrix = [[9] * size, [7] * (size - 1) + [0]]
    bMatrix = [5] * size
    zVector, xVector = [0] * size, [0] * size
    resultVector = []

    startTime = time.time()
    zVector[-1] = bMatrix[size - 1] / bandMatrix[0][size - 1]
    xVector[-1] = 1 / bandMatrix[0][size - 1]

    for index in range(size - 2, -1, -1):
        zVector[index] = (bMatrix[-2] - bandMatrix[1][index] * zVector[index + 1]) / bandMatrix[0][index]
        xVector[index] = (1 - bandMatrix[1][index] * xVector[index + 1]) / bandMatrix[0][index]

    delta = sum(zVector) / (1 + sum(xVector))

    for zVal, xVal in zip(zVector, xVector):
        resultVector.append(zVal - xVal * delta)

    return time.time() - startTime, resultVector


def generate_graph():
    numpyResults = {}
    algorythmResults = {}

    for size in range(100, 20000, 200):
        numpyResults[size] = get_solution_by_numpy_lib(size)[0] * 1000000
        algorythmResults[size] = get_solution_by_sherman_morrison(size)[0] * 1000000

    plt.grid(True)
    plt.yscale('log')
    plt.title('Solving time')
    plt.xlabel('Matrix dimension (N)')
    plt.ylabel('Microseconds (μs)')
    plt.plot(numpyResults.keys(), numpyResults.values(), 'tab:green')
    plt.plot(numpyResults.keys(), algorythmResults.values(), 'tab:red')
    plt.legend(['Solving time by numPy library', 'Solving time by algorythm'])
    plt.show()


if __name__ == '__main__':
    print(f'The solution of the equation Ay = b for N = 50:\n{get_solution_by_sherman_morrison(50)[1]}')
    generate_graph()
