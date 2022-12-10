import matplotlib.pyplot as plt
import sys
from config import X_NEW, PARAMS

def interpolation(fun, node, arg, n):
    x = node(n)
    y, y_new = list(map(lambda a: fun(a), x)), []
    for a in arg:
        val = 0
        # Interpolacja
        for i in range(n + 1):
            tmp = 1
            for k in range(n + 1):
                if i != k:
                    tmp = tmp * (a - x[k]) / (x[i] - x[k])
            val = val + y[i] * tmp
        y_new.append(val)
    return y_new

def savePlot(arg):
    for column, function in enumerate(PARAMS.values()):
        for row, node in enumerate(function['nodes']):
            plt.figure(figsize=(8, 8))
            plt.title(
                'Wielomiany interpolacyjne dla funkcji\n' rf"{function['function_title']} i siatki {node['node_title']}")
            plt.plot(arg, function['function'](arg), 'r', label='f(x)', linewidth=2.5)
            for degree in node['polynomial_degree']:
                plt.plot(arg, interpolation(function['function'], node['function'], arg, degree),
                         label=rf"$W_{'{' + str(degree) + '}'}(x)$")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid()
            plt.legend()
            plt.savefig(f"wykres{column + row}.svg")
            plt.clf()
            return True

def showPlot(arg):
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    for column, function in enumerate(PARAMS.values()):
        for row, node in enumerate(function['nodes']):
            axs[column, row].set_title(
                'Wielomiany interpolacyjne dla funkcji\n' rf"{function['function_title']} i siatki {node['node_title']}")
            axs[column, row].plot(arg, function['function'](arg), 'r', label='f(x)', linewidth=2.5)
            for degree in node['polynomial_degree']:
                axs[column, row].plot(arg, interpolation(function['function'], node['function'], arg, degree),
                                      label=rf"$W_{'{' + str(degree) + '}'}(x)$")
            axs[column, row].legend(loc=8, prop={'size': 8})
            axs[column, row].grid()
    plt.setp(axs[:, :], xlabel='x', ylabel='y')
    fig.tight_layout(pad=1.0)
    plt.show()


if __name__ == "__main__":
    savePlot(X_NEW) if sys.argv[1] == 'save' else showPlot(X_NEW)
