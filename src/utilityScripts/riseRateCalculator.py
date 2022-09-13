# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:50:44 2020

@author: Malachi
"""


import numpy as np
import os

fileToBeInspected = 'W11_L2_1645UTC_081320_Laurens_Profile.txt' #'W5_L2_1820UTC_070220_Laurens_Profile.txt'
microHodoDir = 'microHodographs'     #location where selections from siftThroughUV are saved. This is also the location where do analysis looks for micro hodos to analysis



g = 9.8     #m * s^-2
heightSamplingFreq = 5     #used in determining moving ave filter window, among other things
linesInHeader = 20     #number of lines in header of txt profile
linesInFooter = 10     #num of lines in footer
col_names = ['Time', 'P', 'T', 'Hu', 'Ws', 'Wd', 'Long.', 'Lat.', 'Alt', 'Geopot', 'MRI', 'RI', 'Dewp.', 'VirtTemp', 'Rs', 'Elevation', 'Az', 'Range', 'D']     #header titles


os.chdir("C:/Users/Malachi/OneDrive - University of Idaho/%SummerInternship2020/hodographAnalyzer/hodographAnalysis")


df = np.genfromtxt(fname=fileToBeInspected, skip_header=linesInHeader, skip_footer=linesInFooter, names=col_names)
df = df[0 : np.where(df['Alt']== df['Alt'].max())[0][0]+1]   
#df = df[::heightSamplingFreq]   #Sample data at nth height

global Time 
global Pres 
global Temp 
global Hu 
global Wd 
global Long 
global Lat 
global Alt 
global Geopot 


Time = df['Time']
Pres = df['P'] 
Temp = df['T'] 
Hu = df['Hu']
Ws = df['Ws'] 
Wd = df['Wd'] 
Long = df['Long']
Lat = df['Lat']
Alt = df['Alt'] 
Geopot = df['Geopot']


diff = np.diff(Alt)
print(diff)
mean = np.mean(diff[0:-1])