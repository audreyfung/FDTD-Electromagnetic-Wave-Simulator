# FDTD Electromagnetic Wave Simulator

Python implementation of the **Finite-Difference Time-Domain (FDTD)** method for simulating coupled electromagnetic and axion fields in 1D and 2D.

## Simulation

https://github.com/user-attachments/assets/27a1c0e1-ba29-4744-9094-6dd8c3ff7a61

Example simulation showing the time evolution and propagation of the coupled fields.

## Key Features

- Implemented an explicit **FDTD solver** for coupled field equations
- **1D and 2D numerical simulations** on spatial grids
- Staggered-grid finite-difference discretization
- Time integration subject to the **CFL stability condition**
- Implemented Gaussian pulse and plane-wave source injection
- **NumPy**-based array operations for numerical computation
- **Numba JIT compilation** to accelerate computationally intensive update loops
- Numerical stability analysis and comparison with the uncoupled system
- Visualization and analysis of simulated field evolution

## Technologies

**Python · NumPy · Numba · Matplotlib · Jupyter · FDTD · Numerical PDEs**
