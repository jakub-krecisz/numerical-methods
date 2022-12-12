import matplotlib.pyplot as plt
import sys
from config import X_RANGE, FUNCTION_PARAMS

def interpolation(function, nodeFunction, xArguments, degree):
    x = nodeFunction(degree)
    y, y_new = list(map(lambda a: function(a), x)), []
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

def save_plots(arg):
    for column, function in enumerate(FUNCTION_PARAMS.values()):
        for row, node in enumerate(function['interpolation_nodes']):
            plt.figure(figsize=(8, 8))
            plt.title(
                'Wielomiany interpolacyjne dla funkcji\n' rf"{function['function_title']} i siatki {node['node_title']}")
            plt.plot(arg, function['function'](arg), 'r', label='f(x)', linewidth=2.5)
            for degree in node['polynomial_degrees']:
                plt.plot(arg, interpolation(function['function'], node['function'], arg, degree),
                         label=rf"$W_{'{' + str(degree) + '}'}(x)$")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid()
            plt.legend()
            plt.savefig(f"wykres_{column}-{row}.svg")
            plt.clf()

def show_plots(arg):
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    for column, function in enumerate(FUNCTION_PARAMS.values()):
        for row, node in enumerate(function['interpolation_nodes']):
            axs[column, row].set_title(
                'Wielomiany interpolacyjne dla funkcji\n' rf"{function['function_title']} i siatki {node['node_title']}")
            axs[column, row].plot(arg, function['function'](arg), 'r', label='f(x)', linewidth=2.5)
            for degree in node['polynomial_degrees']:
                axs[column, row].plot(arg, interpolation(function['function'], node['function'], arg, degree),
                                      label=rf"$W_{'{' + str(degree) + '}'}(x)$")
            axs[column, row].legend(loc=8, prop={'size': 8})
            axs[column, row].grid()
    plt.setp(axs[:, :], xlabel='x', ylabel='y')
    fig.tight_layout(pad=1.0)
    plt.show()


if __name__ == "__main__":
    save_plots(X_RANGE) if sys.argv[1] == 'save' else show_plots(X_RANGE)
