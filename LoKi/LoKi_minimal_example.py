#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:54:24 2023

@author: s1984454
"""
from LoKi import LoKi

Psi = 5 #Dimensionless central concentration.
mu = 0 # Dimensionless mass of central object, zero for classical king models.
epsilon = 1e-6 #radius to begin integrating from, can't be zero but can be very small provided that mu is zero.

model = LoKi(mu, epsilon, Psi)

r = model.rhat # Dimensionless radial grid that the solution is evaluated at.
psi = model.psi # Dimensionless escape energy profile.
rho = model.rho_hat # Dimensionless density profile.

print(r)
print(psi)
print(rho)
