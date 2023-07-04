# Derivative Approximation

The program calculates the approximation of derivatives using two different formulas:

a) Dhf(x) = [f(x+h) - f(x)] / h

b) Dhf(x) = [f(x+h) - f(x-h)] / (2h)

It analyzes the behavior of the error |Dhf(x) - f'(x)| for the function f(x) = sin(x) and the point x = 0.2, by varying the parameter h for different floating-point types (float, double). It also plots the |Dhf(x) - f'(x)| in a logarithmic scale. Additionally, you can experiment with other functions like exp and cos.

## Requirements

- Python 3.x
- NumPy
- pandas
- matplotlib

## Configuration

The behavior of the program can be customized using the `config.py` file. The available configuration options are:

- `POINT_VALUE`: The value of the point for which the derivatives are approximated (default: 0.2).
- `POINT_AMOUNT`: The number of points in the h range (default: 200).
- `NUM_OF_ROWS`: The number of rows to print in the table (default: `POINT_AMOUNT/20`).
- `PRECISION`: The precision for each data type, specifying the number of digits after the decimal point. The precision can be customized for different data types, such as 'float32', 'float64', and 'double' (default: {'float32': 7, 'float64': 16, 'double': 16}).
- `FILE_NAME_PLOT`: The file name for the generated plot image (default: 'generated_plots/plot.png').
- `FILE_NAME_TABLE`: The file name for the generated table image, where `{dataType}` will be replaced with the actual data type (default: 'table_{dataType}.png').

Modify the values in the `config.py` file according to your preferences before running the program.

## Usage

Run the program using the command `make <arg>`, where `<arg>` is one of the three available arguments: `plot`, `table_float`, `table_double`. For example, to generate the plot, use `make plot`. To generate the table for the 'float32' data type, use `make table_float`.

## Results

After running the program, you will obtain a plot that shows the mismatch between the discrete derivative and the exact derivative for the function f(x) = sin(x) and the point x = 0.2.

![plot.png](generated_plots%2Fplot.png)
