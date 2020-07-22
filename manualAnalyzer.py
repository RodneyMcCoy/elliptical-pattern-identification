# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 09:56:40 2020

@author: Malachi
"""
import os
import numpy as np
import pandas as pd

#for ellipse fitting
from math import atan2
from numpy.linalg import eig, inv, svd

#data smoothing
import scipy
from scipy import signal


#metpy related dependencies - consider removing entirely
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
from metpy.units import units


#variables
g = 9.8     #m * s^-2
heightSamplingFreq = 5     #used in determining moving ave filter window, among other things
linesInHeader = 20     #number of lines in header of txt profile
linesInFooter = 10     #num of lines in footer
col_names = ['Time', 'P', 'T', 'Hu', 'Ws', 'Wd', 'Long.', 'Lat.', 'Alt', 'Geopot', 'Dewp.', 'VirtTemp', 'Rs', 'D']     #header titles
minAlt = 12000 * units.m     #minimun altitude of analysis
#variables for potential temperature calculation
p_0 = 1000 * units.hPa     #needed for potential temp calculatiion
movingAveWindow = 11     #need to inquire about window size selection

latitudeOfAnalysis = 45

microHodoDir = 'microHodographs'     #location where selections from siftThroughUV are saved. This is also the location where do analysis looks for micro hodos to analyse

def doAnalysis(microHodoDir):
    #query list of potential wave candidates
    
    #specify current working directory
    #os.chdir(microHodoDir)
    print("microPathExists")
    print(os.path.exists(microHodoDir))
    
    for file in os.listdir(microHodoDir):
        path = os.path.join(microHodoDir, file)
        print('Analyzing micro-hodos in: {}'.format(file))
        #print(os.getcwd())
        #x = os.path.isfile(file)
        #print(x)
        df = np.genfromtxt(fname=path, delimiter=',', names=True)
        
        instance = microHodo(df['Alt'], df['u'], df['v'], df['temp'], df['bv2'])
        eps = instance.fit_ellipse()
        print("Ellipse Properties:", eps)
        params = instance.getParameters(eps)
        print("Wave Parameters:", params)
 

    









def getFiles():    
    # specify the location of flight data
    os.chdir("C:/Users/Malachi/OneDrive - University of Idaho/%SummerInternship2020/hodographAnalysis")
    #os.getcwd()



def truncateByAlt(df):
    #df = np.genfromtxt(fname=file, skip_header=linesInHeader, skip_footer=linesInFooter, names=col_names)
    df = df[df['Alt'] >= minAlt.magnitude]
    df = df[0 : np.where(df['Alt']== df['Alt'].max())[0][0]+1]
    
    return df

def preprocessDataNoResample(file):
    # perform calculations on data to prepare for hodograph
    
    # interpret flight data into usable array/dictionary/list (not sure which is preferable yet...)
    df = np.genfromtxt(fname=file, skip_header=linesInHeader, skip_footer=linesInFooter, names=col_names)
    
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
    
    df = truncateByAlt(df)
    
    print("df size")
    print(df.size)
    print(df)
    #print(df)
    
    
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
    
    plt.plot(u, Alt, label='Raw')
    
    # run moving average over u,v comps
    altExtent = max(Alt) - minAlt    #NEED TO VERIFY THE CORRECT WINDOW SAMPLING SZE
    print("Alt Extent")
    print(altExtent)
    window = int((altExtent.magnitude / heightSamplingFreq / 4))    #removed choosing max between calculated window and 11,artifact from IDL code
    print("WINDOW SIZE")
    print(window)
    mask = np.ones(1100) / window
    
    print("Mask")
    print(mask.size)
    print(mask)
    #uMean = np.convolve(u.magnitude, mask, 'same') * units.m/units.second
    uMean = signal.savgol_filter(u.magnitude, window, 3) * units.m/units.second
    vMean = signal.savgol_filter(v.magnitude, window, 3) * units.m/units.second
    tempMean = signal.savgol_filter(Temp.magnitude, window, 3) * units.degC
    

    #uMean = pd.Series(u).rolling(window=window, center=True, min_periods=1).mean().to_numpy() * units.m / units.second #units dropped, rolling ave ccalculated, units added
    #vMean = pd.Series(v).rolling(window=window, center=True, min_periods=1).mean().to_numpy() * units.m / units.second #units dropped, rolling ave ccalculated, units added
    #tempMean = pd.Series(Temp).rolling(window=window, center=True, min_periods=1).mean().to_numpy() * units.degC #units dropped, rolling ave ccalculated, units added
    print("UMean")
    print(uMean)
    plt.plot(uMean, Alt, label='Mean')
    
    #subtract background
    u -= uMean
    v -= vMean
    Temp -= tempMean

    print("meanSmoothedData:")
    print(np.mean(v))
    
    plt.plot(u, Alt, label='Smoothed')
    plt.legend(loc='upper right')
    #print("u comps")
    #print(u)
    #print(type(u))
    
     #various calcs
    global potentialTemp
    global bv2
    potentialTemp = mpcalc.potential_temperature(Pres, Temp).to('degC')    #potential temperature
    bv2 = mpcalc.brunt_vaisala_frequency_squared(Alt, potentialTemp)    #N^2
    
    
       
    #print('bv2')
    #print(bv2)
    #print('pt')
    #print(potentialTemp)
    #potentialTemp = potentialTemperature(Pres, Temp)
    #bvFreqSquared = bruntVaisalaFreqSquared(Alt, potentialTemp)
    
    
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

    
class microHodo:
    def __init__(self, ALT, U, V, TEMP, BV2):
      self.alt = ALT#.magnitude
      self.u = U#.magnitude
      self.v = V#.magnitude
      self.temp = TEMP#.magnitude
      self.bv2 = BV2#.magnitude
      
    def getParameters(self, eps):
        lambda_z = (self.alt[-1] - self.alt[0]) * 2
        m = 2 * np.pi / lambda_z
        phi = eps[4]    #orientation of ellipse
        rot = [[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]
        uv = [self.u, self.v]
        uvrot = rot * uv
        return lambda_z
 
    def saveMicroHodo(self, upperIndex, lowerIndex, fname):
        wd = os.getcwd()
        T = np.column_stack([self.alt[lowerIndex:upperIndex+1], self.u[lowerIndex:upperIndex+1], self.v[lowerIndex:upperIndex+1], self.temp[lowerIndex:upperIndex+1], self.bv2[lowerIndex:upperIndex+1]])
        T = pd.DataFrame(T, columns = ['Alt', 'u', 'v', 'temp', 'bv2'])
        #print("T")
        #print(T)
        fname = '{}-{}-{}'.format(fname, int(self.alt[lowerIndex]), int(self.alt[upperIndex]))
        T.to_csv('{}/microHodographs/{}.csv'.format(wd, fname), index=False, float_format='%4.3f')

    def ellipse_center(self, a):
        """@brief calculate ellipse centre point
        @param a the result of __fit_ellipse
        """
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        num = b * b - a * c
        x0 = (c * d - b * f) / num
        y0 = (a * f - b * d) / num
        return np.array([x0, y0])
    
    
    def ellipse_axis_length(self, a):
        """@brief calculate ellipse axes lengths
        @param a the result of __fit_ellipse
        """
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        down1 = (b * b - a * c) *\
                ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        down2 = (b * b - a * c) *\
                ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        res1 = np.sqrt(up / down1)
        res2 = np.sqrt(up / down2)
        return np.array([res1, res2])
    
    
    def ellipse_angle_of_rotation(self, a):
        """@brief calculate ellipse rotation angle
        @param a the result of __fit_ellipse
        """
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        return atan2(2 * b, (a - c)) / 2
    
    def fmod(self, x, y):
        """@brief floating point modulus
            e.g., fmod(theta, np.pi * 2) would keep an angle in [0, 2pi]
        @param x angle to restrict
        @param y end of  interval [0, y] to restrict to
        """
        r = x
        while(r < 0):
            r = r + y
        while(r > y):
            r = r - y
        return r
    
    
    def __fit_ellipse(self, x,y):
        """@brief fit an ellipse to supplied data points
                    (internal method.. use fit_ellipse below...)
        @param x first coordinate of points to fit (array)
        @param y second coord. of points to fit (array)
        """
        x, y = x[:, np.newaxis], y[:, np.newaxis]
        D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
        S, C = np.dot(D.T, D), np.zeros([6, 6])
        C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
        U, s, V = svd(np.dot(inv(S), C))
        return U[:, 0]
    
    
    def fit_ellipse(self):
        """@brief fit an ellipse to supplied data points: the 5 params
            returned are:
            a - major axis length
            b - minor axis length
            cx - ellipse centre (x coord.)
            cy - ellipse centre (y coord.)
            phi - rotation angle of ellipse bounding box
        @param x first coordinate of points to fit (array)
        @param y second coord. of points to fit (array)
        """
        x, y = self.u, self.v
        e = self.__fit_ellipse(x,y)
        centre, phi = self.ellipse_center(e), self.ellipse_angle_of_rotation(e)
        axes = self.ellipse_axis_length(e)
        a, b = axes
    
        # assert that a is the major axis (otherwise swap and correct angle)
        if(b > a):
            tmp = b
            b = a
            a = tmp
    
            # ensure the angle is betwen 0 and 2*pi
            phi = self.fmod(phi, 2. * np.pi)   #originally alpha = ...
        return [a, b, centre[0], centre[1], phi]

def siftThroughUV(u, v, Alt):

    #plot Altitude vs. U,V in two subplots
    fig1 = plt.figure("U, V, hodo")

    U = plt.subplot(131)
    V = plt.subplot(132)
    microscopicHodo = plt.subplot(133)
    fig1.suptitle('Alt vs u,v')
    U.plot(u, Alt, linewidth=.5)
    V.plot(v, Alt, linewidth=.5)
    fig1.tight_layout()
    
    print("Made it through")
    
    upperIndex, lowerIndex = hodoPicker()
    #plot selected altitude window in third subplot
    microscopicHodo.plot(u[lowerIndex:upperIndex], v[lowerIndex:upperIndex])
    #plt.ioff()
    #fig1.show()
    plt.pause(.1)   #crude solution that forces hodograph to update before user io is queried // what is the elegant solution?
    ellipseSave = "Save Hodograph Data? (y/n/nextFile)"
    string = input(ellipseSave)
    if string == 'n':
        print("Hodo not saved")
        
    elif string == 'y' :
        print("Hodograph saved")
        temporary = microHodo(Alt, u, v, Temp, bv2)
        temporary.saveMicroHodo(upperIndex, lowerIndex, 'fname')
        
        #break
    elif string == nextFile:
        print("Continuing to next file")
    
    print("DONE W LOOP")
    
    
    
    #print(type(result))
    #result.cla()
    
#---------------------------------------------------------------------
"""
''' fit_ellipse.py by Nicky van Foreest '''

def ellipse_center(a):
    @brief calculate ellipse centre point
    @param a the result of __fit_ellipse
    
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])


def ellipse_axis_length(a):
    @brief calculate ellipse axes lengths
    @param a the result of __fit_ellipse

    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) *\
            ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) *\
            ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])


def ellipse_angle_of_rotation(a):
    @brief calculate ellipse rotation angle
    @param a the result of __fit_ellipse
    
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return atan2(2 * b, (a - c)) / 2

def fmod(x, y):
    @brief floating point modulus
        e.g., fmod(theta, np.pi * 2) would keep an angle in [0, 2pi]
    @param x angle to restrict
    @param y end of  interval [0, y] to restrict to
    
    r = x
    while(r < 0):
        r = r + y
    while(r > y):
        r = r - y
    return r


def __fit_ellipse(x, y):
    @brief fit an ellipse to supplied data points
                (internal method.. use fit_ellipse below...)
    @param x first coordinate of points to fit (array)
    @param y second coord. of points to fit (array)
    
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S, C = np.dot(D.T, D), np.zeros([6, 6])
    C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
    U, s, V = svd(np.dot(inv(S), C))
    return U[:, 0]


def fit_ellipse(x, y):
    @brief fit an ellipse to supplied data points: the 5 params
        returned are:
        a - major axis length
        b - minor axis length
        cx - ellipse centre (x coord.)
        cy - ellipse centre (y coord.)
        phi - rotation angle of ellipse bounding box
    @param x first coordinate of points to fit (array)
    @param y second coord. of points to fit (array)
    
    e = __fit_ellipse(x, y)
    centre, phi = ellipse_center(e), ellipse_angle_of_rotation(e)
    axes = ellipse_axis_length(e)
    a, b = axes

    # assert that a is the major axis (otherwise swap and correct angle)
    if(b > a):
        tmp = b
        b = a
        a = tmp

        # ensure the angle is betwen 0 and 2*pi
        phi = fmod(phi, 2. * np.pi)   #originally alpha = ...
    return [a, b, centre[0], centre[1], phi]
"""
#---------------------------------------------------------------------
#Call functions for analysis------------------------------------------
plt.close('all')
getFiles()
preprocessDataNoResample('W5_L2_1820UTC_070220_Laurens_Profile.txt')
#macroHodo()
#siftThroughUV(u, v, Alt)
#eps = fit_ellipse(temporary.u, temporary.v)
#print(eps)
#hodoPicker()
doAnalysis(microHodoDir)

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