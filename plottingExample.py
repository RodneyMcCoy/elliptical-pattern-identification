# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 08:36:40 2020

@author: Malachi
"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


bins = range(1000)

df = pd.DataFrame({'rnd':np.random.binomial(100, .05, 1000), 'bins': range(1000)})
#print(df)

rollingMean1 = df.rnd.rolling(window=10).mean()
rollingMean2 = df.rnd.rolling(window=50).mean()


plt.plot(df.bins, df.rnd, label='rnd')
plt.plot(df.bins, rollingMean1, color='orange')
plt.plot(df.bins, rollingMean2, color='pink')
plt.show()

