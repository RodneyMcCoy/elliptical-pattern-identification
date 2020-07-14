# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 09:56:40 2020

@author: Malachi
"""
import os
import numpy as np
import pandas as pd

import metpy.calc as mpcalc
import matplotlib.pyplot as plt
#from metpy.plots import Hodograph
from metpy.units import units

#variables
g = 9.8     #m * s^-2
heightSamplingFreq = 1     #used in determining moving ave filter window, among other things
linesInHeader = 20     #number of lines in header of txt profile
linesInFooter = 10     #num of lines in footer
col_names = ['Time', 'P', 'T', 'Hu', 'Ws', 'Wd', 'Long.', 'Lat.', 'Alt', 'Geopot', 'Dewp.', 'VirtTemp', 'Rs', 'D']     #header titles
minAlt = 12000 * units.m     #minimun altitude of analysis
#variables for potential temperature calculation
p_0 = 1000 * units.hPa     #needed for potential temp calculatiion
movingAveWindow = 11     #need to inquire about window size selection

latitudeOfAnalysis = 45



select = False     #flag for plotting hodograph after selection
def getFiles():    
    # specify the location of flight data
    os.chdir("C:/Users/Malachi/OneDrive - University of Idaho/%SummerInternship2020/hodographMethod")
    #os.getcwd()



def truncateByAlt(df):
    #df = np.genfromtxt(fname=file, skip_header=linesInHeader, skip_footer=linesInFooter, names=col_names)
    df = df[df['Alt'] >= minAlt.magnitude]
    df = df[0 : np.where(df['Alt']== df['Alt'].max())[0][0]+1]
    
    return df

def preprocessDataNoResample():
    # perform calculations on data to prepare for hodograph
    
    # interpret flight data into usable array/dictionary/list (not sure which is preferable yet...)
    df = np.genfromtxt(fname='W5_L2_1820UTC_070220_Laurens_Profile.txt', skip_header=linesInHeader, skip_footer=linesInFooter, names=col_names)
    
    df = df[::heightSamplingFreq]   #Sample data at nth height
    
    global Time 
    global Pres 
    global Temp 
    global Hu 
    global Wd 
    global Long 
    global Lat 
    global Alt 
    global Geopot 
    
    df = truncateByAlt(df)
    print(df)
    
    
    Time = df['Time']
    Pres = df['P'] * units.hPa
    Temp = df['T'] * units.degC
    Hu = df['Hu']
    Ws = df['Ws'] * units.m / units.second
    Wd = df['Wd'] * units.degree
    Long = df['Long']
    Lat = df['Lat']
    Alt = df['Alt'] * units.meter
    Geopot = df['Geopot']
    
    
    #sampledAlt = Alt[::heightSamplingFreq] # find nth element in list
    global u, v     #make components accessible outside of function
    u, v = mpcalc.wind_components(Ws, Wd)   #raw u,v components

    # run moving average over u,v comps
    altExtent = max(Alt) - minAlt    #NEED TO VERIFY THE CORRECT WINDOW SAMPLING SZE
    print("Alt Extent")
    print(altExtent)
    window = max(int((altExtent.magnitude / heightSamplingFreq / 4)), 11)
    print("WINDOW SIZE")
    print(window)
    uMean = pd.Series(u).rolling(window=window, center=True, min_periods=1).mean().to_numpy() * units.m / units.second #units dropped, rolling ave ccalculated, units added
    vMean = pd.Series(v).rolling(window=window, center=True, min_periods=1).mean().to_numpy() * units.m / units.second #units dropped, rolling ave ccalculated, units added
    TempMean = pd.Series(Temp).rolling(window=10, center=True, min_periods=1).mean().to_numpy() * units.degC #units dropped, rolling ave ccalculated, units added

    
    #various calcs
    global potentialTemp
    global bvFreqSquared
    potentialTemp = mpcalc.potential_temperature(Pres, Temp).to('degC')    #potential temperature
    bvFreqSquared = mpcalc.brunt_vaisala_frequency_squared(Alt, potentialTemp)    #N^2
    
    #potentialTemp = potentialTemperature(Pres, Temp)
    #bvFreqSquared = bruntVaisalaFreqSquared(Alt, potentialTemp)
    
    print('bv2')
    print(bvFreqSquared)
    print('pt')
    print(potentialTemp)
    
        
    #subtract background
    u -= uMean
    v -= vMean
    Temp -= TempMean
 
    print("u")
    print(u)
    print(len(u))
    
def macroHodo():
    #plot v vs. u
    plt.figure("Macroscopic Hodograph")  #Plot macroscopic hodograph
    c=Alt
    plt.scatter( u, v, c=c, cmap = 'magma', s = 1, edgecolors=None, alpha=1)
    cbar = plt.colorbar()
    cbar.set_label('Altitude')        
           
def hodoPicker():
    # #plot Altitude vs. U,V in two subplots
    # fig1 = plt.figure("U, V, hodo")
    # U = plt.subplot(131)
    # V = plt.subplot(132)
    # fig1.suptitle('Alt vs u,v')
    # U.plot(u, Alt, linewidth=.5)
    # V.plot(v, Alt, linewidth=.5)
    # fig1.tight_layout()
    
    
    #Select points from graph, choose indices of data that are closest match to selections
    print('Select Two Altitudes to Examine')
    altitudes =plt.ginput(2, timeout=0)
    alt1, alt2 = [i[1] for i in altitudes]
    print("Alt 1, 2")
    print(alt1, alt2)
    alt1, alt2 = np.argmin(abs(Alt.magnitude - alt1)), np.argmin(abs(Alt.magnitude-alt2))
    upperIndex, lowerIndex = max(alt1, alt2), min(alt1, alt2)     #indices of upper,lower altitudes
    return upperIndex, lowerIndex

    # #plot selected altitude window in third subplot
    # microHodo = plt.subplot(133)
    # microHodo.plot(u[lowerIndex:upperIndex], v[lowerIndex:upperIndex])
    # fig1.tight_layout()
    # fig1.show()
    
def fitEllipse():
    #conic least squares algorithm
    
def doAnalysis():
    #query list of potential wave candidates
    
    

def siftThroughUV():
    #plot Altitude vs. U,V in two subplots
    fig1 = plt.figure("U, V, hodo")
    U = plt.subplot(131)
    V = plt.subplot(132)
    fig1.suptitle('Alt vs u,v')
    U.plot(u, Alt, linewidth=.5)
    V.plot(v, Alt, linewidth=.5)
    fig1.tight_layout()
    microHodo = plt.subplot(133)
    fig1.show()
    while True:
        print("Made it through")
        
        upperIndex, lowerIndex = hodoPicker()
        #plot selected altitude window in third subplot
        print(upperIndex)
        microHodo.plot(u[lowerIndex:upperIndex], v[lowerIndex:upperIndex])
        print("HERE")
        fig1.tight_layout()
        fig1.show()
        ellipseSave = "Save Hodograph Data? (y/n)"
        string = input(ellipseSave)
        if string == 'n':
            microHodo.cla()    
            continue
        else:
            break
    
    
    print("DONE W LOOP")
    
    
    #print(type(result))
    #result.cla()
    
    
#Call functions for analysis------------------------------------------
plt.close('all')
getFiles()
preprocessDataNoResample()
#macroHodo()
siftThroughUV()

#hodoPicker()


#---------------------------------------------------------------------




#Artifacts
"""
#print(u)
#print(u, v)
#print(Wd)
# create hodograph
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
h = Hodograph(ax, component_range=40)
h.add_grid(10)
h.plot(-u, -v)  # need to verify the direction of u,v. There is a 180deg modulo ambiguity, this should be resolved by labeling radial azmiths on plot


#print(u, v)

"""


"""
#Rise rate calculation--------------------------------------------
deltaZ = np.array([]) 
for i in range(len(Alt)):
    if i < len(Alt) - 1:
        delta = Alt[i + 1] - Alt[i]
        deltaZ = np.append(deltaZ, delta)

rr = deltaZ * units.meters / (1 * units.second)
print(rr)
appendedTime = Time[0:-1]
meanRr = rr.mean()
print(meanRr)

plt.plot(appendedTime, rr)
plt.show()
#End rise rate calculation-----------------------------------------
"""
"""
# Function for calculating potential temperature, as done in Tom's code
def potentialTemperature(Pressure, Temperature):
    potTemp = (p_0 ** 0.286) * Temperature / (Pressure ** 0.286)
    return potTemp
#test potential temperature calculator__________________________________________
#print(potentialTemperature(Pres, Temp).to('degC'))
#print(mpcalc.potential_temperature(Pres, Temp).to('degC')) 
#Results from metPy seem to match Tom's function
"""

""" Possibly use to replace metPy calculations   
def bruntVaisalaFreqSquared(alt, potTemp):
    N2 = (g / potTemp) * np.gradient(potTemp, heightSamplingFreq)
    return N2

def potentialTemperature(pres, temp):
    theta = temp * (1000 / pres) ** 0.268
    return theta
"""   


# fit ellipses to hodograph

# use ellipse characteristics to determine wave parameters

# Visualize? 