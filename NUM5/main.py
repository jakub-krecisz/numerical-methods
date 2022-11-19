import numpy as np
import matplotlib.pyplot as plt
import copy
import random

# Imlementacja metody Jacobiego
def jacobi(x, stop):
    err = []
    norm1 = 0
    while stop:
        y = x.copy()
        for i in range(n):
            if i == 0:
                x[i] = (b[i]-x[i+1]-0.2*x[i+2])/3
            elif i == 1:
                x[i] = (b[i] - y[i-1] - x[i+1]-0.2*x[i+2])/3
            elif i == n-2:
                x[i] = (b[i] - y[i-1]-0.2*y[i-2] - x[i+1])/3
            elif i == n-1:
                x[i] = (b[i] - y[i-1] - 0.2*y[i-2])/3
            else:
                x[i] = (b[i] - y[i-1] - 0.2*y[i-2]-x[i+1]-0.2*x[i+2])/3
        norm2 = np.sqrt(sum(map(lambda a, b: (a - b)**2, x, y)))
        err.append(copy.deepcopy(x))

        # Sprawdzenie czy zbiegło
        if abs(norm1-norm2) < 10**(-12):
            break
        norm1 = norm2
        stop = stop-1

    print("Wynik otrzymany metodą Jacobiego:")
    print(x)
    print()
    return err

# Implementacja metody Gaussa-Seidela
def gauss(x, stop):
    err = []
    norm1 = 0
    while stop:
        y = x.copy()
        for i in range(n):
            if i == 0:
                x[i] = (b[i]-x[i+1]-0.2*x[i+2])/3
            elif i == 1:
                x[i] = (b[i] - x[i-1] - x[i+1]-0.2*x[i+2])/3
            elif i == n-2:
                x[i] = (b[i] - x[i-1]-0.2*x[i-2] - x[i+1])/3
            elif i == n-1:
                x[i] = (b[i] - x[i-1] - 0.2*x[i-2])/3
            else:
                x[i] = (b[i] - x[i-1] - 0.2*x[i-2]-x[i+1]-0.2*x[i+2])/3

        norm2 = np.sqrt(sum(map(lambda a, b: (a - b)**2, x, y)))
        err.append(copy.deepcopy(x))

        # Sprawdzenie czy zbiegło
        if abs(norm1-norm2) < 10**(-12):
            break
        norm1 = norm2
        stop = stop-1

    print("Wynik otrzymany metodą Gaussa-Seidela:")
    print(x)
    print()
    return err


n = 100
stop = 500
x = random.sample(range(500), 100)
b = list(range(1, n + 1))


err1 = jacobi(x.copy(), stop)
err2 = gauss(x.copy(), stop)

# Obliczanie różnicy pomiędzy rozwiązaniem w poszczególnych iteracjach a rozwiązaniem dokładnym
w1 = []
last1 = err1[-1]
for i in range(len(err1)-1):
    w1.append(np.sqrt(sum(map(lambda a, b: (a - b)**2, err1[i], last1))))

w2 = []
last2 = err2[-1]
for i in range(len(err2)-1):
    w2.append(np.sqrt(sum(map(lambda a, b: (a - b)**2, err2[i], last2))))


# Tworzenie wykresu
plt.grid(True)
plt.xlabel('n')
plt.ylabel("$|x(n) - x(ostatni)|$")
plt.yscale('log')
plt.plot([i for i in range(1, len(w1)+1)], w1)
plt.plot([i for i in range(1, len(w2)+1)], w2)
plt.legend(['Metoda Jacobiego', 'Metoda Gaussa-Seidela'])
plt.title('Porównanie metod iteracyjnych, start w losowym wektorze x')
plt.show()
