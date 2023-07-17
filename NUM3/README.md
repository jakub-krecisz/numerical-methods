# Solving Linear System of Equations

## Problem Description

We are given a matrix A and a vector x. Our task is to find the vector y = A⁻¹x for the following matrix A:

**A:**
```
1.2     0.1/1   0.4/1^2     ...
0.2     1.2     0.1/2       0.4/2^2     ...
...     0.2     1.2         0.1/3       0.4/3^2     ...
...     ...     ...         ...         ...         ...     ...
                                        1.2         0.1/N   0.4/N^2
```


and the vector x = (1, 2, ..., N)ᵀ. We set N = 100. We also need to calculate the determinant of matrix A. The task should be solved using appropriate methods and by implementing the matrix structure.

## Solution

To avoid computing the inverse of matrix A, we rewrite the equation as follows: A𝑦 = 𝑥. We notice that matrix A is a tridiagonal matrix, which means it only has non-zero values on its diagonal and the diagonals above and below the main diagonal. This structure allows us to simplify the solution method.

We will implement two methods for solving the system of equations:

1. Using the numpy library for solving the system using `np.linalg.solve()`.
2. Implementing numerical methods (Gaussian elimination with partial pivoting) to solve the system.

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Usage

Run the program using the command `python main.py`.

## Results
![computing_time.svg](generated_plots%2Fcomputing_time.svg)
