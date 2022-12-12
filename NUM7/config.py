import numpy as np

def function_y(x):
    return 1 / (1 + 25 * (x ** 2))

def function_g(x):
    return np.arctan(np.exp(x))

def homogeneous_interpolation_node(n):
    return [-1 + 2 * i / n for i in range(n + 1)]

def inhomogeneous_interpolation_node(n):
    return np.cos([((2 * i + 1) / (2 * (n + 1))) * np.pi for i in range(n + 1)])


X_RANGE = np.arange(-1.0, 1.01, 0.01)

FUNCTION_PARAMS = {
    'first_function': {
        'function': function_y,
        'function_title': r'$f(x)=\frac{1}{1+25x^2}$',
        'interpolation_nodes': [
            {'function': homogeneous_interpolation_node, 'node_title': r'$x_i=-1+2\frac{i}{n}$', 'polynomial_degrees': [2, 5, 9, 12, 15]},
            {'function': inhomogeneous_interpolation_node, 'node_title': r'$x_i=\cos(\pi\frac{2i+1}{2(n+1)})$', 'polynomial_degrees': [2, 5, 8, 20, 40]}
        ]
    },
    'second_function': {
        'function': function_g,
        'function_title': r'$f(x)=\frac{1}{1+x^2}$',
        'interpolation_nodes': [
            {'function': homogeneous_interpolation_node, 'node_title': r'$x_i=-1+2\frac{i}{n}$', 'polynomial_degrees': [2, 3, 9, 30, 50]},
            {'function': inhomogeneous_interpolation_node, 'node_title': r'$x_i=\cos(\pi\frac{2i+1}{2(n+1)})$', 'polynomial_degrees': [2, 3, 9, 30, 50]}
        ]
    }
}
