import sys
import sympy
import numpy as np
import matplotlib.pyplot as plt

from config import FUNCTION_A_COMPONENTS, FILE_NAME, FUNCTION_B_COMPONENTS, FUNCTION_B_COEFFICIENTS, \
    FUNCTION_B_ARGUMENTS, NOISE_SCALE, NUM_POINTS, X


def _evaluate_function(function_components: np.ndarray, function_coefficients: np.ndarray,
                       x_args: np.ndarray) -> np.ndarray:
    """
    Evaluates the function by combining its components and coefficients.

    :param function_components: Components of the function.
    :param function_coefficients: Coefficients of the function.
    :param x_args: List of x arguments.
    :return: List of function values for the given x arguments.
    """
    function = sum([coefficient * component for coefficient, component in
                    zip(function_coefficients, function_components)])
    return np.array([function.evalf(subs={X: arg}) for arg in x_args])


def _get_coefficients_using_least_squares_method(given_points: np.ndarray,
                                                 function_components: np.ndarray) -> np.ndarray:
    """
    Determines the coefficients of the function using the least squares method.

    :param given_points: List of points for interpolation.
    :param function_components: Components of the function.
    :return: Vector of coefficients corresponding to each component.
    """
    matrix_A = np.column_stack(list(map(lambda f: [f.evalf(subs={X: x_value}) for x_value in given_points[:, 0]],
                                        function_components))).astype(np.double)
    return np.linalg.inv(matrix_A.T @ matrix_A) @ matrix_A.T @ given_points[:, 1]


def generate_plot_with_points_approximation(points_to_approximate: np.ndarray, function_components: list,
                                            exact_coefficients: np.ndarray = None) -> None:
    """
    Generates a plot of function approximation using the least squares method.

    :param points_to_approximate: Points for interpolation.
    :param function_components: Components of the function.
    :param exact_coefficients: Exact coefficients of the function (optional).
    """
    function_components = np.array(list(map(sympy.sympify, function_components)))
    coefficients = _get_coefficients_using_least_squares_method(points_to_approximate, function_components)
    dense_x_range = np.arange(points_to_approximate[:, 0].min(), points_to_approximate[:, 0].max() + 0.01, 0.01)

    title = "Function approximation using least squares method\n$F(x)=" + \
            "+".join([f"{chr(ord('a') + i)}*{c}" for i, c in enumerate(function_components)]) + r"$"
    if exact_coefficients is not None:
        title += f"\nNumber of points = {NUM_POINTS} | Noise scale = {NOISE_SCALE}"

    plt.figure(figsize=(10, 7))
    plt.xlabel("x"), plt.ylabel("y")
    plt.grid(True)
    plt.title(title.replace('**', '^'))

    print(f"{title.replace('$', '').replace('**', '^')}\nFound coefficients = {coefficients}")
    if exact_coefficients is None:
        plt.plot(points_to_approximate[:, 0], points_to_approximate[:, 1],
                 'o', color='black', label='Exact points', lw=3)
        plt.plot(dense_x_range, _evaluate_function(function_components, coefficients, dense_x_range),
                 color='red', label='Approximation')
    else:
        print(f"Exact coefficients = {exact_coefficients}")
        plt.plot(dense_x_range, _evaluate_function(function_components, exact_coefficients, dense_x_range),
                 color='black', label='Exact function', lw=4)
        plt.plot(dense_x_range, _evaluate_function(function_components, coefficients, dense_x_range),
                 color='red', label='Approximation')

        # plot points with added noise for each y value
        # plt.plot(points_to_approximate[:, 0], points_to_approximate[:, 1], 'o', color='green',
        #          label='Points with noise')
    plt.legend(loc='lower center')
    plt.show() if sys.argv[2] == 'show' else plt.savefig(f"generated_plots/{sys.argv[1]}_plot.svg")


if __name__ == '__main__':
    if sys.argv[1] == 'point_a':
        generate_plot_with_points_approximation(np.loadtxt(FILE_NAME), FUNCTION_A_COMPONENTS)
    elif sys.argv[1] == 'point_b':
        y_values = _evaluate_function(np.array(list(map(sympy.sympify, FUNCTION_B_COMPONENTS))),
                                      FUNCTION_B_COEFFICIENTS, FUNCTION_B_ARGUMENTS)
        y_values = y_values + np.random.normal(0, NOISE_SCALE, len(y_values))
        points = np.array([[x, y] for x, y in zip(FUNCTION_B_ARGUMENTS, y_values)], dtype=np.double)
        generate_plot_with_points_approximation(points, FUNCTION_B_COMPONENTS, FUNCTION_B_COEFFICIENTS)
