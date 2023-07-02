# Function Approximation using Least Squares Method

This program performs function approximation using the least squares method. It allows you to find the best coefficients that describe a given set of points and generate graphical plots for the approximations.

## Problem Description

Given a set of points represented by their x and y coordinates, we model these points using the function F(x) = a * sin(2x) + b * sin(3x) + c * cos(5x) + d * exp(-x).

(a) Find the optimal values for the coefficients a-d that best describe the given data using the least squares method. Present the results graphically. In solving this problem, you cannot use library functions for approximation. However, you can use procedures from linear algebra.

(b) Propose a new function G(x) (dependent on multiple parameters) and generate a set of points in the form (x, G(x) + δy), where δy represents random disturbances. Repeat the approximation process from part (a) for the new data and check if the previously determined parameter values can be reconstructed. Experiment with different numbers of generated points and magnitudes of disturbances.

## Requirements

To run this program, the following dependencies are required:

- Python (version 3.6 or higher)
- NumPy
- SymPy
- matplotlib

## Installation

1. Clone the repository or download the program files.
2. Ensure that you have the listed dependencies installed on your system.
3. Run the program using the command `make <arg>`, where `<arg>` is one of the four available arguments: `a_show`, `a_save`, `b_show`, `b_save`. For example, `make a_show`.

## Argument Description

- `a_show`: Shows the approximation of original function.
- `a_save`: Saves generated approximation of original function to svg file.
- `b_show`: Shows the approximation of distributed function.
- `b_save`: Saves generated approximation of distributed function to svg file.

## Configuration

The configuration for the approximation can be found in the `config.py` file. You can customize the following parameters:

- `FUNCTION_A_COMPONENTS`: Components of Function A.
- `FUNCTION_B_COMPONENTS`: Components of Function B.
- `FUNCTION_B_COEFFICIENTS`: Coefficients of Function B.
- `FUNCTION_B_ARGUMENTS`: Arguments of Function B.
- `NOISE_SCALE`: Noise scale.
- `NUM_POINTS`: Number of points for approximation.
- `X`: Symbol x.

## Plots

### Original Function Approximation

The plot below shows the approximation for the original function.

![point_a_plot.svg](generated_plots%2Fpoint_a_plot.svg)

### Disturbed Function Approximation

The plot below shows the approximation for the function with disturbances.

![point_b_plot.svg](generated_plots%2Fpoint_b_plot.svg)
