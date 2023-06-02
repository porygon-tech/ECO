#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 17:30:08 2023

@author: roman
"""
import numpy as np
def pM (zdiffs, alpha=50):
    return np.exp(-alpha*(zdiffs)**2)

def convpM(values,nstates,alpha):
    c = np.zeros((nstates))
    for i in range(nstates):
        c = c + pM(np.arange(nstates)-i, alpha)*values[i]
    return c




