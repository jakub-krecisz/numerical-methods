import sys
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from config import X_RANGE, FUNCTION_PARAMS

def interpolation(function, nodeFunction, xArguments, degree):
    x = nodeFunction(degree)
    y, y_new = list(map(function, x)), []
    for xArgument in xArguments:
        val = 0
        for j in range(degree + 1):
            product = 1
            for k in range(degree + 1):
                if j != k:
                    product *= (xArgument - x[k]) / (x[j] - x[k])
            val += y[j] * product
        y_new.append(val)
    return y_new
def save_plots(figure):
    figure.savefig(
        "wykres0_0.svg",
        bbox_inches=mtransforms.Bbox([[0, 0.5], [0.5, 1]]).transformed(figure.transFigure - figure.dpi_scale_trans)
    )
    figure.savefig(
        "wykres0_1.svg",
        bbox_inches=mtransforms.Bbox([[0.5, 0.5], [1, 1]]).transformed(figure.transFigure - figure.dpi_scale_trans)
    )
    figure.savefig(
        "wykres1_0.svg",
        bbox_inches=mtransforms.Bbox([[0, 0], [0.5, 0.5]]).transformed(figure.transFigure - figure.dpi_scale_trans)
    )
    figure.savefig(
        "wykres1_1.svg",
        bbox_inches=mtransforms.Bbox([[0.5, 0], [1, 0.5]]).transformed(figure.transFigure - figure.dpi_scale_trans)
    )

def generate_plots(whatToDo, xArguments):
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(10)
    fig.set_figwidth(16)
    for column, function in enumerate(FUNCTION_PARAMS.values()):
        for row, node in enumerate(function['interpolation_nodes']):
            axs[column, row].set_title('Interpolation polynomials\n'
                                       rf"function: {function['function_title']}, node: {node['node_title']}")
            axs[column, row].plot(xArguments, function['function'](xArguments), 'black',
                                  label=f'{function["function_title"][1]}(x)', linewidth=4)
            for degree in node['polynomial_degrees']:
                axs[column, row].plot(xArguments,
                                      interpolation(function['function'], node['function'], xArguments, degree),
                                      label=rf"$W_{'{' + str(degree) + '}'}(x)$")
            axs[column, row].legend(loc=8, prop={'size': 8})
            axs[column, row].grid()
    plt.setp(axs[:, :], xlabel='x', ylabel='y')
    fig.tight_layout(pad=1.0)
    plt.show() if whatToDo == 'show' else save_plots(fig)

if __name__ == "__main__":
    generate_plots(sys.argv[1], X_RANGE)
