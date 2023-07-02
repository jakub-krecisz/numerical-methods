import numpy as np

# Functions on which we will perform interpolation
def function_y(x):
    return 1 / (1 + 25 * (x ** 2))

def function_g(x):
    return -x / (1 + 100*(x ** 8))

def function_h(x):
    return 1 / (1 + x**4)

def homogeneous_interpolation_node(n):
    return np.linspace(-1, 1, n+1)

def inhomogeneous_interpolation_node(n):
    return np.cos([((2 * i + 1) / (2 * (n + 1))) * np.pi for i in range(n + 1)])


X_RANGE = np.arange(-1.0, 1.01, 0.01)

FUNCTION_CONFIG = {
    'first_function': {
        'function': function_y,
        'function_title': r'$y(x)=\frac{1}{1+25x^2}$',
        'interpolation_nodes': [
            {'function': homogeneous_interpolation_node, 'node_title': r'$x_i=-1+2\frac{i}{n}$', 'polynomial_degrees': [3, 4, 9, 10, 13, 15]},
            {'function': inhomogeneous_interpolation_node, 'node_title': r'$x_i=\cos(\pi\frac{2i+1}{2(n+1)})$', 'polynomial_degrees': [3, 4, 9, 8, 15, 30]}
        ]
    },
    'second_function': {
        'function': function_g,
        'function_title': r'$g(x)=\frac{-x}{1+100x^8}$',
        'interpolation_nodes': [
            {'function': homogeneous_interpolation_node, 'node_title': r'$x_i=-1+2\frac{i}{n}$', 'polynomial_degrees': [3, 5, 8, 11, 13, 15]},
            {'function': inhomogeneous_interpolation_node, 'node_title': r'$x_i=\cos(\pi\frac{2i+1}{2(n+1)})$', 'polynomial_degrees': [3, 5, 7, 10, 20, 60]}
        ]
    }
}
