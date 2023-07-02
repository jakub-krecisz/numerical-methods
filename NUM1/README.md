# Derivative Approximation

This program calculates the approximation of derivatives using two different formulas:

a) Dhf(x) = [f(x+h) - f(x)] / h

b) Dhf(x) = [f(x+h) - f(x-h)] / (2h)


It analyzes the behavior of the error |Dhf(x) - f'(x)| for the function f(x) = sin(x) and the point x = 0.2, by varying the parameter h for different floating-point types (float, double). It also plots the |Dhf(x) - f'(x)| in a logarithmic scale. Additionally, you can experiment with other functions like exp and cos.

## Requirements

- Python 3.x
- NumPy
- pandas
- matplotlib

## Usage

Run the program using the command `make <arg>`, where `<arg>` is one of the three available arguments: `plot`, `table_float`, `table_double`. For example, `make plot`.


## Results

After running the program, you will obtain a plot that shows the mismatch between the discrete derivative and the exact derivative for the function f(x) = sin(x) and the point x = 0.2.

![plot.png](generated_plots%2Fplot.png)
