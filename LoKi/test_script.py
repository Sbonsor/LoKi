#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:35:38 2022

@author: s1984454
"""
from LoKi import LoKi
from LoKi_samp import LoKi_samp
import numpy as np
import matplotlib.pyplot as plt
from loaded_king import LoadedKing
from scipy.interpolate import interp1d
plt.rc('text', usetex=True)

m = LoKi(0.1,0.1,5)

#b = LoadedKing(0.1,0.1,5,'K',1e9,False)

samples = LoKi_samp(m, N=10000, plot = True)

## Validation figures for sampling
fig1,fig2 = samples.print_validation_figs()
fig1.show()
fig2.show()

