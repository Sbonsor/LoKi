#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:35:38 2022

@author: s1984454
"""
from LoKi import LoKi
import numpy as np
import matplotlib.pyplot as plt
from loaded_king import LoadedKing
plt.rc('text', usetex=True)

test_model = LoKi(0.1,0.1,5)

b = LoadedKing(0.1,0.1,5,'K',1e9,False)

fig,ax = plt.subplots(1,1)
ax.plot(test_model.rhat,test_model.psi)
ax.set_xlabel('$\\hat{r}$')
ax.set_ylabel('$\\psi$')
ax.set_yscale('log')
ax.set_xscale('log')