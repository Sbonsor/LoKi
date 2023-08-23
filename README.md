# Loaded King models
A Python code for the solution of the Loaded King models described in (paper link to be added), a script to generate the asymptotic solutions described in this paper, along with a sampler to generate discrete samples from a given model.

## LoKi

### Model parameters
- mu: dimensionless black hole mass (required).
- epsilon: dimensionless inner radius (required).
- Psi: central concentration (required).

### Options
- max_r: Maximum radius to integrate the solution to Poisson's equation up to (Optional, default = 1e9).
- ode_atol: Absolute tolerance for the ODE solver (Optional, default = 1e-10).
- ode_rtol: Relative tolerance for the ODE solver (Optional, default = 1e-8).
- pot_only: Set to True in order to only return the potential and it's gradient for a faster solution (Optional, default = False).
- model: Flag to indicate which model type to solve (Optional, default = 'K')
  - 'K' = King model.
  - 'PT' = Prendergast & Tomer model.
  - 'W' = Wilson model.

### Output
- psi, dpsi_dr, rhat : dimensionless escape energy, it's gradient, and radius all in dimensionless units.
- rt: Truncation radius where psi=0 in dimensionless units.
- rho_hat, P_hat: density and pressure in dimensionless units.
- U_r,M_r,K_r: radial profiles of potential energy, mass, and kinetic energy in dimensionless units.
- U_hat,M_hat,K_hat: Total potential energy, mass, and kinetic energy in dimensionless units.
## Sampler

### Input

- model: A LoKi object of the model you wish to sample (Required).
- N: The number of samples to generate (Required).

### Options
- scale: Set to True in order to scale output into Hénon units (Optional, default = True).
- plot: set to True (along with scale) to produce validation figures for the radial and energy distirbution of particles.

### Output
- A .csv file containing for each particle:
  - x, y, z: Position of the particle in the specified unit system.
  - vx,vy,vz: Velocity of the particle in the specified unit system.
  - m: Mass of the particle in the specified unit system.
- If plot = True:
  - Figure object showing a histogram of radial samples and the expected distribution, see test_script for an example of how to plot these.
  - Figure object showing a histogram of energy samples and the expected distribution.
