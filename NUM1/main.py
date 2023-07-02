import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *


def right_approx(h_points: np.ndarray[float]) -> np.ndarray[float]:
    """
    Computes the right derivative approximation for the given h points.

    :param h_points: Array of h values.
    :return: Array of right derivative approximations.
    """
    return (np.sin(POINT_VALUE + h_points) - np.sin(POINT_VALUE)) / h_points


def central_approx(h_points: np.ndarray[float]) -> np.ndarray[float]:
    """
    Computes the central derivative approximation for the given h points.

    :param h_points: Array of h values.
    :return: Array of central derivative approximations.
    """
    return (np.sin(POINT_VALUE + h_points) - np.sin(POINT_VALUE - h_points)) / (2 * h_points)


def get_diff(function, h_points: np.ndarray[float]) -> np.ndarray[float]:
    """
    Computes the absolute difference between the exact derivative and the derivative approximation.

    :param function: Derivative approximation function.
    :param h_points: Array of h values.
    :return: Array of absolute differences.
    """
    return np.abs(function(h_points) - np.cos(POINT_VALUE))


def draw_subplot(subplot, d_type: str) -> None:
    """
    Draws a subplot of the mismatch between the discrete derivative and the exact derivative.

    :param subplot: Subplot object.
    :param d_type: Data type ('float32' or 'double').
    """
    subplot.grid(True)
    subplot.set_title(f'Mismatch in {d_type} type')

    h_points = np.logspace(-PRECISION[d_type], 0, num=POINT_AMOUNT + 1, dtype=d_type)
    central_diff = np.array(get_diff(central_approx, h_points), dtype=d_type)
    right_diff = np.array(get_diff(right_approx, h_points), dtype=d_type)

    subplot.loglog(h_points, central_diff, 'tab:green')
    subplot.loglog(h_points, right_diff, 'tab:blue')
    subplot.legend(['Central Derivative', 'Right Derivative'])


def generate_plot() -> None:
    """
    Generates a plot showing the mismatch between the discrete derivative and the exact derivative.
    """
    fig, axs = plt.subplots(2)
    fig.suptitle('Mismatch between discrete derivative and exact derivative\n'
                 f'Function: sin(x) Mismatch in point x={POINT_VALUE}')
    fig.set_size_inches(7, 10)
    for ax in axs.flat:
        ax.set(xlabel="h", ylabel="$|D_hf(x) - f'(X)|$")

    draw_subplot(axs[0], 'float32')
    draw_subplot(axs[1], 'double')
    plt.savefig(FILE_NAME_PLOT, dpi=300)


def print_table(d_type: str) -> None:
    """
    Prints a table showing the mismatch between the discrete derivative and the exact derivative.

    :param d_type: Data type ('float32' or 'double').
    """
    h_points = np.logspace(-PRECISION[d_type], 0, num=POINT_AMOUNT + 1)
    central_diff = np.array(get_diff(central_approx, h_points))
    right_diff = np.array(get_diff(right_approx, h_points))

    df = pd.DataFrame(data={'H': h_points,
                            'Central Derivative difference': central_diff,
                            'Right Derivative difference': right_diff})

    with pd.option_context('display.float_format', lambda x: f'{x:,.{PRECISION[d_type]}f}'):
        print(df[::int(NUM_OF_ROWS)])


if __name__ == '__main__':
    generate_plot()
    if sys.argv[1] == 'plot':
        generate_plot()
    elif sys.argv[1] == 'table':
        print_table(sys.argv[2])
    else:
        print(f'Bad argument! - [plot/table] instead of: {sys.argv[1]}')
        sys.exit()
