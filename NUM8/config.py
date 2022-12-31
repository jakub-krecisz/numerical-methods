import sympy
import numpy as np


# name of the file from which we are getting dots
FILE_NAME = 'data_NUM8.dat'
# our symbol of argument in function
X = sympy.Symbol('x')
# function_a in the form of a list of components, with + by default between elements
FUNCTION_A_COMPONENTS = ['sin(2*x)', 'sin(3*x)', 'cos(5*x)', 'exp(-x)']
# poczatek i koniec przedzialu x
X_START = 0
X_STOP = 10
# Liczba punktów do wygenerowania
NUM_POINTS = 100
# Skala zaburzeń
NOISE_SCALE = 0.000001
# Funkcja B
FUNCTION_B_COMPONENTS = ['cos(2*x)', 'sin(3*x)', 'cos(5*x)', 'exp(-x)']
FUNCTION_B_COEFFICIENTS = np.array([1, 2, 3, 4])
FUNCTION_B_XPOINTS = np.linspace(X_START, X_STOP, NUM_POINTS)
