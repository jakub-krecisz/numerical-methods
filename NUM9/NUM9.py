import numpy as np
import matplotlib.pyplot as plt


# Funckja pierwsza
def funA(x):
    return np.sin(x) - 0.37


# Pochodna funkcji pierwszej
def dfA(x):
    return np.cos(x)


# Funkcja druga
def funB(x):
    return (np.sin(x) - 0.37) ** 2


# Pochodna funkcji drugiej
def dfB(x):
    return 2 * np.cos(x) * (np.sin(x) - 0.37)


# Funkcja usprawniająca
def funU(x):
    return (np.sin(x) - 0.37) / (2 * np.cos(x))


# Pochodna funkcji usprawniającej
def dfU(x):
    return (100 - 37 * np.sin(x)) / (200 * (np.cos(x) ** 2))


# Funkcja obliczająca miejsce zerowe metodą bisekcji
def bisection(f, a, b, e, n, root):
    wynik = []

    # Obliczanie miejsca zerowego
    for i in range(n):
        c = (a + b) / 2  # Punkt środkowy przedziału
        wynik.append(abs(c - root))
        fc = f(c)

        # Zawężenie przedziałów
        if (f(a) * fc) < 0:
            b = c
        elif (fc * f(b)) < 0:
            a = c

        # Warunek stopu
        if not abs(root - c) > e:
            break

    return wynik


# Funkcja obliczająca miejsce zerowe metodą Falsi
def falsi(f, a, b, e, n, root):
    wynik = []

    # Obliczanie miejsca zerowego
    for i in range(n):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))  # Punkt na prostej łączącej a, b dla któego wartość = 0
        wynik.append(abs(c - root))
        fc = f(c)

        # Zawężenie przedziałów
        if (f(a) * fc) < 0:
            b = c
        elif (fc * f(b)) < 0:
            a = c

        # Warunek stopu
        if not abs(root - c) > e:
            break

    return wynik


# Funkcja obliczająca miejsce zerowe metodą siecznych
def sieczne(f, x1, x0, e, n, root):
    wynik = []

    # Obliczanie miejsca zerowego
    for i in range(n):
        wynik.append(abs(x1 - root))

        # Warunek stopu
        if not abs(root - x1) > e:
            break

        # Zabezpieczenie przed dzieleniem przez 0
        if x1 == x0:
            return

        x2 = (x0 * f(x1) - x1 * f(x0)) / (f(x1) - f(x0))  # Nowy punkt

        # Zmiana przedziałów
        x0 = x1
        x1 = x2

    return wynik


# Funkcja obliczająca miejsce zerowe metodą Newtona
def newton(f, df, x, e, n, root):
    wynik = []

    # Obliczanie miejsca zerowego
    for i in range(n):
        wynik.append(abs(x - root))

        # Warunek stopu
        if not abs(root - x) > e:
            break

        x = x - (f(x) / df(x))  # Nowy punkt

    return wynik


if __name__ == "__main__":
    PI2 = np.pi / 2
    e = 10 ** (-6)
    root = np.arcsin(0.37)
    n = 1000

    # Wykres dla pierwszej funkcji
    wb = bisection(funA, 0, PI2, e, n, root)
    print("Pierwiastek funkcji f(x) metodą bisekcji = ", wb[-1])
    plt.plot([i for i in range(len(wb))], wb, '-o', label='Metoda bisekcji')
    wb = falsi(funA, 0, PI2, e, n, root)
    print("Pierwiastek funkcji f(x) metodą Falsi = ", wb[-1])
    plt.plot([i for i in range(len(wb))], wb, '-o', label='Metoda Falsi')
    wb = sieczne(funA, 0, PI2, e, n, root)
    print("Pierwiastek funkcji f(x) metodą siecznych = ", wb[-1])
    plt.plot([i for i in range(len(wb))], wb, '-o', label='Metoda siecznych')
    wb = newton(funA, dfA, 0, e, n, root)
    print("Pierwiastek funkcji f(x) metodą Newtona = ", wb[-1])
    plt.plot([i for i in range(len(wb))], wb, '-o', label='Metoda Newtona')
    plt.grid()
    plt.legend()
    plt.title('Porównanie metod szukania miejsca zerowego\nfunkcji f(x) = sin(x) - 0.37')
    plt.yscale('log')
    plt.show()

    print()
    plt.clf()

    # Wykres dla drugiej funkcji
    # Funkcja nie zmienia znaku więc metoda bisekcji i Falsi nie działą
    wb = sieczne(funB, 0, PI2, e, n, root)
    print("Pierwiastek funkcji g(x) metodą siecznych = ", wb[-1])
    plt.plot([i for i in range(len(wb))], wb, '-o', label='Metoda siecznych')
    wb = newton(funB, dfB, 0, e, n, root)
    print("Pierwiastek funkcji g(x) metodą Newtona = ", wb[-1])
    plt.plot([i for i in range(len(wb))], wb, '-o', label='Metoda Newtona')
    plt.grid()
    plt.legend()
    plt.title('Porównanie metod szukania miejsca zerowego\nfunkcji g(x) = $(sin(x) - 0.37)^2$')
    plt.yscale('log')
    plt.show()

    print()
    plt.clf()

    # Wykres dla usprawnienia
    wb = sieczne(funU, 0, PI2, e, n, root)
    print("Pierwiastek funkcji u(x) metodą siecznych = ", wb[-1])
    plt.plot([i for i in range(len(wb))], wb, '-o', label='Metoda siecznych')
    wb = newton(funU, dfU, 0, e, n, root)
    print("Pierwiastek funkcji u(x) metodą Newtona = ", wb[-1])
    plt.plot([i for i in range(len(wb))], wb, '-o', label='Metoda Newtona')
    plt.grid()
    plt.legend()
    plt.title('Porównanie metod szukania miejsca zerowego\n' + r'funkcji u(x) = $\frac{g(x)}{g\backprime(x)}$')
    plt.yscale('log')
    plt.show()
