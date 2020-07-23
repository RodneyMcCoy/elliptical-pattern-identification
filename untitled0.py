# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:34:25 2020

@author: Malachi
"""


from scipy import integrate
import matplotlib.pyplot as plt
>>>
x = np.linspace(-2, 2, num=20)
y = x
y_int = integrate.cumtrapz(y, x, initial=0)
plt.plot(x, y_int, 'ro', x, y[0] + 0.5 * x**2, 'b-')
plt.show()