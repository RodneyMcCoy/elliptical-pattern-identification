# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:48:06 2021

@author: Malachi
"""
import numpy as np
import matplotlib.pyplot as plt


xc = 0
yc = 0
a = 1
b = .5
theta = -np.pi/4
param = np.linspace(0, 2 * np.pi)    
x = a * np.cos(param) * np.cos(theta) - b * np.sin(param) * np.sin(theta) + xc
y = a * np.cos(param) * np.sin(theta) + b * np.sin(param) * np.cos(theta) + yc

fig,graph = plt.subplots()
graph.plot(x, y, color='red')
graph.set_xlim((-2,2))
graph.set_ylim((-2,2))
plt.show()