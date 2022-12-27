import sys
import sympy
import numpy as np
import matplotlib.pyplot as plt

from config import FUNCTION_A_COMPONENTS, FILE_NAME, FUNCTION_B_COMPONENTS, FUNCTION_B_COEFFICIENTS,\
    FUNCTION_B_XPOINTS, NOISE_SCALE, X


def _evaluate_function(function_components, function_coefficients, x_args):
    """

    :param function_components:
    :param function_coefficients:
    :param x_args:
    :return:
    """
    function = sum([coefficient * component for coefficient, component in
                    zip(function_coefficients, function_components)])
    return [function.evalf(subs={X: arg}) for arg in x_args]


def _get_coefficients_using_least_squares_method(given_points, function_components):
    """

    :param given_points:
    :param function_components:
    :return:
    """
    AMatrix = np.column_stack(list(map(lambda f: [f.evalf(subs={X: x_value}) for x_value in given_points[:, 0]],
                                       function_components))).astype(np.double)
    return np.linalg.inv(AMatrix.T @ AMatrix) @ AMatrix.T @ given_points[:, 1]


def generate_plot_with_points_approximation(points_to_approximate, function_components, exact_coefficients=None):
    """

    :param points_to_approximate:
    :param function_components:
    :param exact_coefficients:
    :return:
    """
    function_components = list(map(sympy.sympify, function_components))
    coefficients = _get_coefficients_using_least_squares_method(points_to_approximate, function_components)
    dense_x_range = np.arange(points_to_approximate[:, 0].min(), points_to_approximate[:, 0].max() + 0.01, 0.01)

    plt.figure(figsize=(10, 7))
    plt.xlabel("x"), plt.ylabel("y")
    plt.grid(True)
    plt.title("Function approximation using least squares method\n$F(x)=" +
              "+".join([f"{chr(ord('a') + i)}*{c}" for i, c in enumerate(function_components)]) + r"$")

    if exact_coefficients is None:
        plt.plot(points_to_approximate[:, 0], points_to_approximate[:, 1], 'o', color='black', lw=3)
        plt.plot(dense_x_range, _evaluate_function(function_components, coefficients, dense_x_range), 'tab:red')
    else:
        plt.plot(dense_x_range, _evaluate_function(function_components, exact_coefficients, dense_x_range), color='green')
        plt.plot(dense_x_range, _evaluate_function(function_components, coefficients, dense_x_range), color='black')
    plt.show()


if __name__ == '__main__':
    if sys.argv[1] == 'point_a':
        generate_plot_with_points_approximation(np.loadtxt(FILE_NAME), FUNCTION_A_COMPONENTS)
    elif sys.argv[1] == 'point_b':
        # TODO: cos jest nie tak, gdy zmienisz poczatkowa funkcje to aproxymacja nie dziala, sprawdz to
        yValues = _evaluate_function(list(map(sympy.sympify, FUNCTION_A_COMPONENTS)), FUNCTION_B_COEFFICIENTS,
                                     FUNCTION_B_XPOINTS)
        yValues = yValues + np.random.normal(0, NOISE_SCALE, len(yValues))
        points = np.array([[x, y] for x, y in zip(FUNCTION_B_XPOINTS, yValues)], dtype=np.double)
        generate_plot_with_points_approximation(points, FUNCTION_B_COMPONENTS, FUNCTION_B_COEFFICIENTS)
