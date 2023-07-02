import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from config import X_RANGE, FUNCTION_CONFIG


def interpolation(function: callable, node_function: callable, x_arguments: list, degree: int) -> list:
    """
    Interpolates a function using polynomial interpolation.

    :param function: The function to be interpolated.
    :param node_function: The function to generate interpolation nodes.
    :param x_arguments: List of x values for evaluation.
    :param degree: Degree of the interpolation polynomial.
    :return: List of interpolated function values for the given x values.
    """
    x = node_function(degree)
    y, y_new = list(map(function, x)), []
    for x_argument in x_arguments:
        val = 0
        for j in range(degree + 1):
            product = 1
            for k in range(degree + 1):
                if j != k:
                    product *= (x_argument - x[k]) / (x[j] - x[k])
            val += y[j] * product
        y_new.append(val)
    return y_new


def save_plots(figure: plt.plot) -> None:
    """
    Saves the generated plots in SVG format.

    :param figure: The figure object containing the plots.
    """
    figure.savefig(
        "generated_plots/chart0_0.svg",
        bbox_inches=transforms.Bbox([[0, 0.5], [0.5, 1]]).transformed(figure.transFigure - figure.dpi_scale_trans)
    )
    figure.savefig(
        "generated_plots/chart0_1.svg",
        bbox_inches=transforms.Bbox([[0.5, 0.5], [1, 1]]).transformed(figure.transFigure - figure.dpi_scale_trans)
    )
    figure.savefig(
        "generated_plots/chart1_0.svg",
        bbox_inches=transforms.Bbox([[0, 0], [0.5, 0.5]]).transformed(figure.transFigure - figure.dpi_scale_trans)
    )
    figure.savefig(
        "generated_plots/chart1_1.svg",
        bbox_inches=transforms.Bbox([[0.5, 0], [1, 0.5]]).transformed(figure.transFigure - figure.dpi_scale_trans)
    )


def generate_plots(x_arguments: list) -> None:
    """
    Generates and displays plots for polynomial interpolation.

    :param x_arguments: List of x values for evaluation.
    """
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(10)
    fig.set_figwidth(16)
    for column, function in enumerate(FUNCTION_CONFIG.values()):
        for row, node in enumerate(function['interpolation_nodes']):
            axs[column, row].set_title('Interpolation polynomials\n'
                                       rf"function: {function['function_title']}, node: {node['node_title']}")
            axs[column, row].plot(x_arguments, function['function'](x_arguments), 'black',
                                  label=f'{function["function_title"][1]}(x)', linewidth=4)
            for degree in node['polynomial_degrees']:
                axs[column, row].plot(x_arguments,
                                      interpolation(function['function'], node['function'], x_arguments, degree),
                                      label=rf"$W_{'{' + str(degree) + '}'}(x)$")
            axs[column, row].legend(loc=8, prop={'size': 8})
            axs[column, row].grid()
    plt.setp(axs[:, :], xlabel='x', ylabel='y')
    fig.tight_layout(pad=1.0)
    save_plots(fig)
    plt.show()


if __name__ == "__main__":
    generate_plots(X_RANGE)
