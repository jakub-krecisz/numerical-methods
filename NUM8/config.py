import sympy
import numpy as np

# plik z punktami
FILE_NAME = 'data_NUM8.dat'
# our symbol of argument in function
X = sympy.Symbol('x')
# function_a in the form of a list of components, with + by default between elements
FUNCTION_A_COMPONENTS = ['sin(2*x)', 'sin(3*x)', 'cos(5*x)', 'exp(-x)']
# poczatek i koniec przedzialu x
X_START = 0
X_STOP = 10
# Liczba punktów do wygenerowania
NUM_POINTS = 1000
# Skala zaburzeń
NOISE_SCALE = 0.1
# Funkcja B
FUNCTION_B_COMPONENTS = ['sin(3*x)', 'cos(5*x)', 'exp(-x)']
FUNCTION_B_COEFFICIENTS = np.array([6, 5, 4])
FUNCTION_B_XPOINTS = np.linspace(X_START, X_STOP, NUM_POINTS)
