# FDTD Electromagnetic Wave Simulator

A Python implementation of the Finite-Difference Time-Domain (FDTD) method for simulating axion-modified Maxwell equations. The project numerically evolves coupled electromagnetic and axion fields in one and two dimensions and investigates photon–axion conversion and numerical stability.

## Overview

Axions are hypothetical particles that can interact with electromagnetic fields through an $E \cdot B$ coupling. This project implements the resulting modified Maxwell equations as a numerical time-domain simulation.

The electromagnetic equations are reformulated in terms of transformed electric and magnetic fields, while the axion field is evolved using its coupled wave equation. The physical electromagnetic fields are then recovered from the transformed fields and axion field.

The numerical equations are discretized using finite differences on a spatial grid and evolved explicitly in time using an FDTD scheme.


## Example Simulation

https://github.com/user-attachments/assets/27a1c0e1-ba29-4744-9094-6dd8c3ff7a61

The animation below shows the evolution of the simulated fields during photon–axion conversion.

## Features

- 1D FDTD solver for coupled electromagnetic and axion fields
- Extension to 2D electromagnetic field simulations
- Gaussian pulse and plane-wave sources
- External magnetic-field injection
- Evolution of the axion field alongside electromagnetic fields
- Conversion between transformed and physical electromagnetic fields
- Numba-accelerated numerical update loops
- Visualization of electromagnetic and axion field propagation
- Numerical stability testing under the CFL condition
- Comparison between axion-coupled and standard electromagnetic simulations

## Results

### 1D photon–axion conversion

The 1D simulations demonstrate photon–axion conversion when an external magnetic field is applied parallel to the incident electric field. The axion field is generated after the electromagnetic wave enters the external magnetic-field region and subsequently propagates through the simulation domain.

The simulations also verify the expected dependence on the $E \cdot B$ coupling: conversion is suppressed when the external magnetic field is removed or oriented perpendicular to the incident electric field.

### 1D plane-wave simulation

A plane-wave configuration was also simulated using parameters motivated by solar photons and a strong external magnetic field. The resulting axion amplitude is many orders of magnitude smaller than the electromagnetic field, illustrating the weak photon–axion conversion expected in this regime.

### 2D simulations

The FDTD implementation was extended to two spatial dimensions to investigate electromagnetic and axion-field propagation in a more general geometry.

The 2D simulations also highlighted numerical challenges associated with localized source injection. In particular, magnetic-field divergence near a point source can lead to numerical instabilities in the axion-coupled calculation.

A plane-wave source provides a more stable configuration, while a Total-Field/Scattered-Field (TFSF) source treatment and improved absorbing boundary conditions were identified as possible approaches for further development.

## Numerical Method

The simulations use an explicit finite-difference time-domain scheme. Spatial and temporal derivatives are approximated using finite differences, with electromagnetic fields updated on a staggered grid.

The time step is chosen to satisfy the Courant–Friedrichs–Lewy (CFL) stability condition. For the simulations presented here,

$$
\frac{\Delta t}{\Delta x} = 0.5.
$$

The 1D and 2D implementations use NumPy arrays for the field variables and Numba JIT compilation to accelerate the computationally intensive grid-update loops.

## Technologies

- Python
- NumPy
- Matplotlib
- Numba
- Jupyter Notebook
- Finite-Difference Time-Domain (FDTD) methods
- Numerical PDE simulation

