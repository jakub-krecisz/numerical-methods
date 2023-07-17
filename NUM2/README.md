# Linear System of Equations

## Problem Description

The given coefficient matrices are:

**A1**:
```
 2.34332898  -0.11253278  -0.01485349   0.33316649   0.71319625
-0.11253278   1.67773628  -0.32678856  -0.31118836  -0.43342631
-0.01485349  -0.32678856   2.66011353   0.85462464   0.16698798
 0.33316649  -0.31118836   0.85462464   1.54788582   0.32269197
 0.71319625  -0.43342631   0.16698798   0.32269197   3.27093538
```

**A2**:
```
 2.34065520  -0.05353743   0.00237792   0.32944082   0.72776588
-0.05353743   0.37604149  -0.70698859  -0.22898376  -0.75489595
 0.00237792  -0.70698859   2.54906441   0.87863502   0.07309288
 0.32944082  -0.22898376   0.87863502   1.54269444   0.34299341
 0.72776588  -0.75489595   0.07309288   0.34299341   3.19154447
```


And the given vectors are:

**b** ≡ (3.55652063354463, −1.86337418741501, 5.84125684808554, −1.74587299057388, 0.84299677124244)<sup>T</sup>

**b'** ≡ b + (10<sup>-5</sup>, 0, 0, 0, 0)<sup>T</sup>

## Requirements

- Python 3.x
- NumPy

## Configuration

**A1_MATRIX**: Coefficient matrix A1, as a NumPy array.
**A2_MATRIX**: Coefficient matrix A2, as a NumPy array.
**B_VECTOR**: Right-hand side vector B, as a NumPy array.
**B_VECTOR_PRIM**: B_VECTOR + (10<sup>-5</sup>, 0, 0, 0, 0)<sup>T</sup>

## Usage

Run the program using the command `make run`.

## Solution

The program uses the numpy library to solve the linear system of equations. It provides the following solutions:

**A₁y₁ = b**:
   - **y₁** = [solution vector]

**A₁y₁' = b'**:
   - **y₁'** = [solution vector]

**A₂y₂ = b**:
   - **y₂** = [solution vector]

**A₂y₂' = b'**:
   - **y₂'** = [solution vector]

## Results and Interpretation

The program calculates the differences between the solution vectors, given by **∆₁ = ||y₁ - y₁'||₂** and **∆₂ = ||y₂ - y₂'||₂**. It also computes the condition numbers of the coefficient matrices, given by **cond(A₁)** and **cond(A₂)**. Here are the results:

- **∆₁** = [value of ∆₁]
- **∆₂** = [value of ∆₂]
- **cond(A₁)** = [condition number of A₁]
- **cond(A₂)** = [condition number of A₂]
