import numpy as np

from config import A1, A2, B

if __name__ == '__main__':
    Bp = B + np.array([10 ** -5, 0, 0, 0, 0]).transpose()

    y1 = np.linalg.solve(A1, B)
    y2 = np.linalg.solve(A2, B)
    yp1 = np.linalg.solve(A1, Bp)
    yp2 = np.linalg.solve(A2, Bp)

    print("y1: ", np.linalg.solve(A1, B))
    print("y2: ", np.linalg.solve(A2, B))

    print("y'1: ", np.linalg.solve(A1, Bp))
    print("y'2: ", np.linalg.solve(A2, Bp))

    d1 = np.linalg.norm(y1 - yp1)
    d2 = np.linalg.norm(y2 - yp2)

    print("delta1: {:.20f}".format(d1))
    print("delta2: {:.20f}".format(d2))
    print('\n')

    # Dodatkowy kod do wniosku
    print("Wspolczynnik uwarunkowania macierzy A1: ", np.linalg.cond(A1))
    print("Wspolczynnik uwarunkowania macierzy A2: ", np.linalg.cond(A2))
