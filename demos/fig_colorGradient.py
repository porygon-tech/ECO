#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:31:46 2023

@author: ubuntu
"""

import matplotlib.pyplot as plt
import numpy as np

# Define matrix dimensions
rows = 101
cols = 101

# Create a matrix where the green values increase with rows and magenta values increase with columns
green_increase = np.linspace(1, 0, rows).reshape(-1, 1)
magenta_increase = np.linspace(1, 0, cols)

# Create a 3-channel color matrix
color_matrix = np.zeros((rows, cols, 3))
color_matrix[:, :, 1] = green_increase  # Green channel
color_matrix[:, :, 2] = magenta_increase  # Magenta channel
color_matrix[:, :, 0] = magenta_increase  # Magenta channel

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
imsh = ax.imshow(color_matrix,origin='lower')
fig.canvas.toolbar_visible = False
fig.canvas.header_visible = False
fig.canvas.resizable = True
plt.show()
