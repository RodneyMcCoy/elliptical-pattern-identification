# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:57:11 2020

@author: Malachi
"""

#dependencies
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate




#Variables related to generatinging array
linesInHeader = 20 # number of lines in header of txt profile
linesInFooter = 10 # num of lines in footer
col_names = ['Time', 'P', 'T', 'Hu', 'Ws', 'Wd', 'Long.', 'Lat.', 'Alt', 'Geopot', 'MRI', 'RI', 'Dewp.', 'VirtTemp', 'Rs', 'Elevation', 'Azimuth', 'Range', 'D']


# specify the location of flight data
os.chdir("C:/Users/Malachi/OneDrive - University of Idaho/%SummerInternship2020/hodographMethod")


def truncateByAlt(file):
    df = np.genfromtxt(fname=file, skip_header=linesInHeader, skip_footer=linesInFooter, names=col_names)
    df = df[0 : np.where(df['Alt']== df['Alt'].max())[0][0]+1]
    
    return df
    
    
df = truncateByAlt('W5_L2_1820UTC_070220_Laurens_Profile.txt')

plt.plot(df['D'], df['Alt'])
plt.xlabel('Density kg * m^-3')
plt.ylabel('Altitude m')

z = np.polyfit(df['D'], df['Alt'],0)
plt.plot(z)
print('Z')
print(z)
plt.show()


