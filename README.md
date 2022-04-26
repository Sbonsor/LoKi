# Loaded King models
A Python code for the solution of the Loaded King models described in (paper here), along with a discrete sampler to generate 

## LoKi

### Model parameters
- mu: dimensionless black hole mass (required).
- epsilon: dimensionless inner radius (required).
- Psi: central concentration (required).

### Options
- scale: Set to True in order to scale output into Hénon units (Optional, default = False).
- project: Set to True in order to provide density and velocity dispersion profiles in projection (Optional, default = False, not implemented yet). 
- max_r: Maximum radius to integrate the solution to Poisson's equation up to (Optional, default = 1e9).
- ode_atol: Absolute tolerance for the ODE solver (Optional, default = 1e-10).
- ode_rtol: Relative tolerance for the ODE solver (Optional, default = 1e-8).
- pot_only: Set to True in order to only return the potential and it's gradient for a faster solution (Optional, default = False).
- asymptotics: Set to True in order to return the asymptotic solutions in each of the three regimes (Optional, default = False, can be set to True only in model == 'K').
- model: Flag to indicate which model type to solve (Optional, default = 'K')
  - 'K' = King model.
  - 'PT' = Prendergast & Tomer model.
  - 'W' = Wilson model.

### Output
- psi, dpsi_dr, rhat : dimensionless escape energy, it's gradient, and radius all in dimensionless units.
- density, pressure: density and pressure in dimensionless units.
- If scale == True:
  - a, A, E_0 : Dimensional constants required for the scaling.
- If project == True:
  - d, proj_density, proj_pressure : projected radius, density, and pressure in dimensionless units.
- If asymptotics == True:
  - psi_X_Y, r_X_Y: dimensionless escape energy and radius in regime X, region Y.
  - C_Psi, D_Psi: Values of C(\Psi), and D(\Psi).

## Sampler

### Input

- model: A LoKi object of the model you wish to sample (Required).
- N: The number of samples to generate (Required).

### Options
- scale: Set to True in order to scale output into Hénon units (Optional, default = True).

### Output
- A .csv file containing for each particle:
  - x, y, z: Position of the particle in the specified unit system.
  - vx,vy,vz: Velocity of the particle in the specified unit system.
  - m: Mass of the particle in the specified unit system.
- U: The total potential energy of the samples.
- K: The total kinetic energy of the samples.
- Virial: The virial ratio as a check on the procedure.
