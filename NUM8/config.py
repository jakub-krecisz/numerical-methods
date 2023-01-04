import sympy
import numpy as np

# Our symbol of argument in function
X = sympy.Symbol('x')

# --- FUNCTION A ---
# Function in the form of a list of components, with + by default between elements
FUNCTION_A_COMPONENTS = ['sin(2*x)', 'sin(3*x)', 'cos(5*x)', 'exp(-x)']

# The file containing the set of points to be approximated
FILE_NAME = 'data_NUM8.dat'


# --- FUNCTION B ---
# Function in the form of a list of components, with + by default between elements
FUNCTION_B_COMPONENTS = ['log(x)', 'cos(x)', 'sin(x)**2', 'sin(5*x)']

# Exact coefficients based on which we will try to approximate with random noise
FUNCTION_B_COEFFICIENTS = [2, -5, 3, 3]

# Number of points to be generated
NUM_POINTS = 100
# Range of x arguments
X_START = 0.01
X_STOP = 10
FUNCTION_B_ARGUMENTS = np.linspace(X_START, X_STOP, NUM_POINTS)

# Scale of random noise for every y value
NOISE_SCALE = 1
