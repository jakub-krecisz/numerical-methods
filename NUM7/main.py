import numpy as np  # Biblioteka numeryczna
import matplotlib.pyplot as plt  # Biblioteka do tworzenia wykresow
import sys  # Biblioteka pozwalajaca obsluge argumentow linii polecen


# Funkcja pierwsza
def fun1(x):
    return 1 / (1 + 25 * (x ** 2))


# Funkcja druga
def fun2(x):
    return 1 / (1 + (x ** 2))


# Funkcja obliczająca węzły jednorodne
def node1(n):
    return [-1 + 2 * i / n for i in range(n + 1)]


# Funkcja obliczająca węzły niejednorodne
def node2(n):
    return np.cos([((2 * i + 1) / (2 * (n + 1))) * np.pi for i in range(n + 1)])


# Funkcja obliczająca wartości wielomianu
def interpolation(fun, node, arg, n):
    x = node(n)
    y = list(map(lambda x: fun(x), x))

    y_new = []
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


# Funkcja tworząca i zapisująca wykresy
def savePlot(arg):
    plt.figure(figsize=(8, 8))
    plt.title('Wielomiany interpolacyjne dla funkcji\n' r'$f(x)=\frac{1}{1+25x^2}$ i siatki $x_i=-1+2\frac{i}{n}$')
    plt.plot(arg, fun1(arg), 'r', label='f(x)', linewidth=2.5)
    plt.plot(arg, interpolation(fun1, node1, arg, 2), 'b', label=r'$W_2(x)$')
    plt.plot(arg, interpolation(fun1, node1, arg, 5), 'g', label=r'$W_5(x)$')
    plt.plot(arg, interpolation(fun1, node1, arg, 9), 'c', label=r'$W_9(x)$')
    plt.plot(arg, interpolation(fun1, node1, arg, 12), 'm', label=r'$W_{12}(x)$')
    plt.plot(arg, interpolation(fun1, node1, arg, 15), 'y', label=r'$W_{15}(x)$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.savefig("wykres1.svg")

    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.title(
        'Wielomiany interpolacyjne dla funkcji\n' r'$f(x)=\frac{1}{1+25x^2}$ i siatki $x_i=\cos(\pi\frac{2i+1}{2(n+1)})$')
    plt.plot(arg, fun1(arg), 'r', label='f(x)', linewidth=2.5)
    plt.plot(arg, interpolation(fun1, node2, arg, 2), 'b', label=r'$W_2(x)$')
    plt.plot(arg, interpolation(fun1, node2, arg, 5), 'g', label=r'$W_5(x)$')
    plt.plot(arg, interpolation(fun1, node2, arg, 8), 'c', label=r'$W_8(x)$')
    plt.plot(arg, interpolation(fun1, node2, arg, 20), 'm', label=r'$W_{20}(x)$')
    plt.plot(arg, interpolation(fun1, node2, arg, 40), 'y', label=r'$W_{40}(x)$')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.savefig("wykres2.svg")

    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.title('Wielomiany interpolacyjne dla funkcji\n' r'$f(x)=\frac{1}{1+x^2}$ i siatki $x_i=-1+2\frac{i}{n}$')
    plt.plot(arg, fun2(arg), 'r', label='f(x)', linewidth=2.5)
    plt.plot(arg, interpolation(fun2, node1, arg, 2), 'b', label=r'$W_2(x)$')
    plt.plot(arg, interpolation(fun2, node1, arg, 3), 'g', label=r'$W_3(x)$')
    plt.plot(arg, interpolation(fun2, node1, arg, 9), 'c', label=r'$W_9(x)$')
    plt.plot(arg, interpolation(fun2, node1, arg, 30), 'm', label=r'$W_{30}(x)$')
    plt.plot(arg, interpolation(fun2, node1, arg, 50), 'y', label=r'$W_{50}(x)$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.savefig("wykres3.svg")

    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.title(
        'Wielomiany interpolacyjne dla funkcji\n' r'$f(x)=\frac{1}{1+x^2}$ i siatki $x_i=\cos(\pi\frac{2i+1}{2(n+1)})$')
    plt.plot(arg, fun2(arg), 'r', label='f(x)', linewidth=2.5)
    plt.plot(arg, interpolation(fun2, node2, arg, 2), 'b', label=r'$W_2(x)$')
    plt.plot(arg, interpolation(fun2, node2, arg, 3), 'g', label=r'$W_3(x)$')
    plt.plot(arg, interpolation(fun2, node2, arg, 9), 'c', label=r'$W_9(x)$')
    plt.plot(arg, interpolation(fun2, node2, arg, 30), 'm', label=r'$W_{30}(x)$')
    plt.plot(arg, interpolation(fun2, node2, arg, 50), 'y', label=r'$W_{50}(x)$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.savefig("wykres4.svg")


# Fukcja tworząca wykres z 4 wykresami
def showPlot(arg):
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    axs[0, 0].set_title(
        'Wielomiany interpolacyjne dla funkcji\n' r'$f(x)=\frac{1}{1+25x^2}$ i siatki $x_i=-1+2\frac{i}{n}$')
    axs[0, 0].plot(arg, fun1(arg), 'r', label='f(x)', linewidth=2.5)
    axs[0, 0].plot(arg, interpolation(fun1, node1, arg, 2), 'b', label=r'$W_2(x)$')
    axs[0, 0].plot(arg, interpolation(fun1, node1, arg, 5), 'g', label=r'$W_5(x)$')
    axs[0, 0].plot(arg, interpolation(fun1, node1, arg, 9), 'c', label=r'$W_9(x)$')
    axs[0, 0].plot(arg, interpolation(fun1, node1, arg, 12), 'm', label=r'$W_{12}(x)$')
    axs[0, 0].plot(arg, interpolation(fun1, node1, arg, 15), 'y', label=r'$W_{15}(x)$')
    axs[0, 0].legend(loc=8, prop={'size': 8})
    axs[0, 0].grid()

    axs[0, 1].set_title(
        'Wielomiany interpolacyjne dla funkcji\n' r'$f(x)=\frac{1}{1+25x^2}$ i siatki $x_i=\cos(\pi\frac{2i+1}{2(n+1)})$')
    axs[0, 1].plot(arg, fun1(arg), 'r', label='f(x)', linewidth=2.5)
    axs[0, 1].plot(arg, interpolation(fun1, node2, arg, 2), 'b', label=r'$W_2(x)$')
    axs[0, 1].plot(arg, interpolation(fun1, node2, arg, 5), 'g', label=r'$W_5(x)$')
    axs[0, 1].plot(arg, interpolation(fun1, node2, arg, 8), 'c', label=r'$W_8(x)$')
    axs[0, 1].plot(arg, interpolation(fun1, node2, arg, 20), 'm', label=r'$W_{20}(x)$')
    axs[0, 1].plot(arg, interpolation(fun1, node2, arg, 40), 'y', label=r'$W_{40}(x)$')
    axs[0, 1].legend(loc=8, prop={'size': 8})
    axs[0, 1].grid()

    axs[1, 0].set_title(
        'Wielomiany interpolacyjne dla funkcji\n' r'$f(x)=\frac{1}{1+x^2}$ i siatki $x_i=-1+2\frac{i}{n}$')
    axs[1, 0].plot(arg, fun2(arg), 'r', label='f(x)', linewidth=2.5)
    axs[1, 0].plot(arg, interpolation(fun2, node1, arg, 2), 'b', label=r'$W_2(x)$')
    axs[1, 0].plot(arg, interpolation(fun2, node1, arg, 3), 'g', label=r'$W_3(x)$')
    axs[1, 0].plot(arg, interpolation(fun2, node1, arg, 9), 'c', label=r'$W_9(x)$')
    axs[1, 0].plot(arg, interpolation(fun2, node1, arg, 30), 'm', label=r'$W_{30}(x)$')
    axs[1, 0].plot(arg, interpolation(fun2, node1, arg, 50), 'y', label=r'$W_{50}(x)$')
    axs[1, 0].legend(loc=8, prop={'size': 8})
    axs[1, 0].grid()

    axs[1, 1].set_title(
        'Wielomiany interpolacyjne dla funkcji\n' r'$f(x)=\frac{1}{1+x^2}$ i siatki $x_i=\cos(\pi\frac{2i+1}{2(n+1)})$')
    axs[1, 1].plot(arg, fun2(arg), 'r', label='f(x)', linewidth=2.5)
    axs[1, 1].plot(arg, interpolation(fun2, node2, arg, 2), 'b', label=r'$W_2(x)$')
    axs[1, 1].plot(arg, interpolation(fun2, node2, arg, 3), 'g', label=r'$W_3(x)$')
    axs[1, 1].plot(arg, interpolation(fun2, node2, arg, 9), 'c', label=r'$W_9(x)$')
    axs[1, 1].plot(arg, interpolation(fun2, node2, arg, 30), 'm', label=r'$W_{30}(x)$')
    axs[1, 1].plot(arg, interpolation(fun2, node2, arg, 50), 'y', label=r'$W_{50}(x)$')
    axs[1, 1].legend(loc=8, prop={'size': 8})
    axs[1, 1].grid()

    plt.setp(axs[:, :], xlabel='x')
    plt.setp(axs[:, :], ylabel='y')
    fig.tight_layout(pad=1.0)
    plt.show()


def main():
    # Sprawdzenie ilosci argumentow
    if len(sys.argv) != 2:
        print("Niepoprawna ilosc argumentow")
        sys.exit()

    x_new = np.arange(-1.0, 1.01, 0.01)

    # Wybor działania
    if sys.argv[1] == 'save':
        savePlot(x_new)
    elif sys.argv[1] == 'show':
        showPlot(x_new)
    else:
        print("Zly argument")
        sys.exit()


if __name__ == "__main__":
    main()
