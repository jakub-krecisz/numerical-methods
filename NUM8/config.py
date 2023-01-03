import sympy
import numpy as np

# The file containing the set of points to be approximated
FILE_NAME = 'data_NUM8.dat'

# Our symbol of argument in function
X = sympy.Symbol('x')

# Function A in the form of a list of components, with + by default between elements
FUNCTION_A_COMPONENTS = ['sin(2*x)', 'sin(3*x)', 'cos(5*x)', 'exp(-x)']

# FUNCTION B
# Number of points to be generated
NUM_POINTS = 10

# Range of x arguments
X_START = 0
X_STOP = 10
FUNCTION_B_XPOINTS = np.linspace(X_START, X_STOP, NUM_POINTS)

# Scale of random noise for every y value
NOISE_SCALE = 1

# Function B in the form of a list of components, with + by default between elements
FUNCTION_B_COMPONENTS = ['sin(2*x)', 'sin(3*x)', 'cos(5*x)', 'exp(-x)']

# Exact coefficients based on which we will try to approximate with random noise
FUNCTION_B_COEFFICIENTS = np.array([1, 2, 3, 4])


