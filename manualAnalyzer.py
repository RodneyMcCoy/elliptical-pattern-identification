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

#manual GUI
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.widgets as widgets




#variables
fileToBeInspected = 'W9_L1_1500UTC_072920_Laurens_Profile.txt'#'W5_L2_1820UTC_070220_Laurens_Profile.txt'
g = 9.8     #m * s^-2
heightSamplingFreq = 5     #used in determining moving ave filter window, among other things
linesInHeader = 20     #number of lines in header of txt profile
linesInFooter = 10     #num of lines in footer
col_names = ['Time', 'P', 'T', 'Hu', 'Ws', 'Wd', 'Long.', 'Lat.', 'Alt', 'Geopot', 'MRI', 'RI', 'Dewp.', 'VirtTemp', 'Rs', 'Elevation', 'Az', 'Range', 'D']     #header titles
minAlt = 1000 * units.m     #minimun altitude of analysis
#variables for potential temperature calculation
p_0 = 1000 * units.hPa     #needed for potential temp calculatiion
movingAveWindow = 11     #need to inquire about window size selection

latitudeOfAnalysis = 45 * units.degree

microHodoDir = 'microHodographs'     #location where selections from siftThroughUV are saved. This is also the location where do analysis looks for micro hodos to analyse

def doAnalysis(microHodoDir):
    #query list of potential wave candidates
    
    #specify current working directory
    #os.chdir(microHodoDir)
    print("microPathExists")
    print(os.path.exists(microHodoDir))
    
    hodo_list = []
    for file in os.listdir(microHodoDir):
        path = os.path.join(microHodoDir, file)
        print('Analyzing micro-hodos in: {}'.format(file))
        
        #print(os.getcwd())
        #x = os.path.isfile(file)
        #print(x)
        df = np.genfromtxt(fname=path, delimiter=',', names=True)
        
        #instanceName = "{}-{}".format(min(df['Alt']), max(df['Alt']))
        instance = microHodo(df['Alt'], df['u'], df['v'], df['temp'], df['bv2'])
        instance.addFileName(file)  #file name added to object attribute here to be used in labeling plots
        instance.addAltitudeCharacteristics()
        
        instance.fit_ellipse()
        
        #print("Ellipse Properties:", eps)
        instance.getParameters()
        hodo_list.append(instance)  #add micro-hodo to running list
        #print("Wave Parameters:", params)
    hodo_list.sort(key=lambda x: x.altOfDetection)  #sort list of hodographs on order of altitude  
    print("HODOLIST", hodo_list)
    return hodo_list     #list of micro-hodograph objects created 
    
def plotBulkMicros(hodo_list, fname):
    #plots all micro-hodographs for a single flight
    bulkPlot = plt.figure(fname)
    plt.suptitle("Micro-hodographs for \n {}".format(fname))#, y=1.09)
    
    
    totalPlots = len(hodo_list)
    numColumns = np.ceil(np.sqrt(totalPlots)).astype('int')
    numRows = np.ceil((totalPlots / numColumns)).astype('int')
    position = range(1, totalPlots + 1)
    print(numRows)
    print("Type; ", type(numRows))
    
    i = 0   #counter for indexing micro-hodo objects
    #k=1 #counter for indexing subplots
    for hodo in hodo_list:
        print("HODO ITERATION: ", hodo)
        ax = bulkPlot.add_subplot(numRows, numColumns, position[i])
        ax.plot(hodo_list[i].u, hodo_list[i].v) 
        
        
        #plot parametric best fit ellipse
        param = np.linspace(0, 2 * np.pi)
        x = hodo_list[i].a * np.cos(param) * np.cos(hodo_list[i].phi) - hodo_list[i].b * np.sin(param) * np.sin(hodo_list[i].phi) + hodo_list[i].c_x
        y = hodo_list[i].a * np.cos(param) * np.sin(hodo_list[i].phi) + hodo_list[i].b * np.sin(param) * np.cos(hodo_list[i].phi) + hodo_list[i].c_y
        ax.plot(x, y)
        
        
        ax.set_title("{}-{} (m)".format(hodo_list[i].lowerAlt, hodo_list[i].upperAlt), fontsize=14 )
        i += 1
        #k += 1
        
    
    
    #plt.tight_layout()
    plt.subplots_adjust(top=.9, hspace=.5)
                  
    plt.show() 
    
    
    








def getFiles():    
    # specify the location of flight data
    os.chdir("C:/Users/Malachi/OneDrive - University of Idaho/%SummerInternship2020/hodographAnalyzer/hodographAnalysis")
    #os.getcwd()



def truncateByAlt(df):
    #df = np.genfromtxt(fname=file, skip_header=linesInHeader, skip_footer=linesInFooter, names=col_names)
    df = df[df['Alt'] > minAlt.magnitude]
    df = df[0 : np.where(df['Alt']== df['Alt'].max())[0][0]+1]
    
    return df

def bruntViasalaFreqSquared(potTemp, heightSamplingFreq):
    G = 9.8 * units.m / units.second**2
    N2 = (G / potTemp) * np.gradient(potTemp, heightSamplingFreq * units.m) 
    return N2

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
    
    
    #various calcs
    tempK = Temp.to('kelvin')
    global potentialTemp
    global bv2
    potentialTemperature =  tempK * (p_0 / Pres) ** (.286)    #try (2/7) for exponent    
    
    #bv2 = mpcalc.brunt_vaisala_frequency_squared(Alt, potTemp)    #N^2
    bv2 = bruntViasalaFreqSquared(potentialTemperature, heightSamplingFreq)
    print("BV2 ", bv2)
    
    
    #sampledAlt = Alt[::heightSamplingFreq] # find nth element in list
    global u, v     #make components accessible outside of function
    u, v = mpcalc.wind_components(Ws, Wd)   #raw u,v components
    

    #--------------------------------------------------------------------------------------
    
    plt.plot(u.magnitude, Alt.magnitude, label='Raw')
    
    # run moving average over u,v comps
    altExtent = max(Alt) - minAlt    #NEED TO VERIFY THE CORRECT WINDOW SAMPLING SZE
    print("Alt Extent")
    print(altExtent)
    window = int((altExtent.magnitude / heightSamplingFreq / 4))    #removed choosing max between calculated window and 11,artifact from IDL code
    if (window%2) == 0:
        window = window-1
        
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
    #--------------------------------------------------------------------------------------
    plt.plot(uMean.magnitude, Alt.magnitude, label='Mean')
    
    #subtract background
    u -= uMean
    v -= vMean
    Temp -= tempMean

    print("meanSmoothedData:")
    print(np.mean(v))
    
    #------------------------------------------------------------------------------------
    plt.plot(u.magnitude, Alt.magnitude, label='Smoothed')
    plt.xlabel('(m/s)', fontsize=40)
    plt.ylabel('(m)', fontsize=40)
    plt.title('Smoothed U Components', fontsize=45)
    plt.legend(loc='upper right', fontsize=30)
    
    
    
    
       
    #print('bv2')
    #print(bv2)
    #print('pt')
    #print(potentialTemp)
    #potentialTemp = potentialTemperature(Pres, Temp)
    #bvFreqSquared = bruntVaisalaFreqSquared(Alt, potentialTemp)
    

    
def macroHodo():
    #plot v vs. u
    plt.figure("Macroscopic Hodograph", figsize=(10,10))  #Plot macroscopic hodograph
    c=Alt
    #plt.scatter( u, v, c=c, cmap = 'magma', s = 1, edgecolors=None, alpha=1)
    plt.plot(u,v)
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
    
def manualGUI():
    
    

    fig5 = plt.figure('MANUAL GUI')
    ax = plt.axes([.25, .25, .6, .6])
    plt.subplots_adjust(left=0.2, bottom=0.2)
    alt0 = 0
    wind0 = 100
    
    l, = plt.plot(u[:alt0+wind0], v[:alt0+wind0], 'o', ls='-', markevery=[0])
    ax.margins(.05)
    plt.axis('equal')
    
    axAlt = plt.axes([0.01, 0.01, 0.02, 0.9])
    axFineAlt = plt.axes([0.05, 0.01, 0.02, 0.9])
    axamp = plt.axes([0.09, 0.01, 0.02, 0.9])
    axoutput = plt.axes([.15, .9, .1, .1])
    axSave = plt.axes([.5, .9, .1, .06])
    
    altSlider = Slider(axAlt, 'Altitude', 0, len(Alt), valinit=wind0, orientation='vertical')
    fineAltSlider = Slider(axFineAlt, 'Fine Altitude \n Adjust', -50, 50, valinit=0, orientation='vertical')
    altWindow = Slider(axamp, 'Window', 0, 1000, valinit=wind0, orientation='vertical')
    t1 = axoutput.text(0, .5 , "Lower Altitude: {}".format(Alt[alt0]))
    saveButton = Button(axSave, "SaveMicro", color='lightgoldenrodyellow', hovercolor='0.975')
    axoutput.axis('off')
    
    
   
    
    
    
    
    def update(val):
        
        sliderAlt = int(altSlider.val) + int(fineAltSlider.val)
        sliderWindow = int(altWindow.val)
        l.set_ydata(v[sliderAlt:sliderAlt+sliderWindow])
        l.set_xdata(u[sliderAlt:sliderAlt+sliderWindow])
        
        #plt.plot(u[sliderAlt], v[sliderAlt], marker='o', color='g', label='point')
        ax.autoscale(enable=True)
        ax.relim()

        t1.set_text("Lower Altitude: {}".format(Alt[sliderAlt]))
        
        fig5.canvas.draw_idle()
        plt.pause(.1)
        saveButton.on_clicked(save)
        
    def save(event):
        altInd = int(altSlider.val) + int(fineAltSlider.val)
        winLength = int(altWindow.val)
        
        ALT = Alt[altInd:altInd+winLength+1]
        U = u[altInd:altInd+winLength+1]
        V = v[altInd:altInd+winLength+1]
        TEMP = Temp[altInd:altInd+winLength+1]
        BV2 = bv2[altInd:altInd+winLength+1]
        instance = microHodo(ALT, U, V, TEMP, BV2)
        instance.addFileName(fileToBeInspected)
        instance.saveMicroHodoNoIndices()
        return
    def saave(event):
        print("SAVED")
    
    
    altSlider.on_changed(update)
    altWindow.on_changed(update)
    fineAltSlider.on_changed(update)
    
    #plt.pause(.1)
    #fig5.show()   needed in working gui
    
    

   
    
class microHodo:
    def __init__(self, ALT, U, V, TEMP, BV2):
      self.alt = ALT#.magnitude
      self.u = U#.magnitude
      self.v = V#.magnitude
      self.temp = TEMP#.magnitude
      self.bv2 = BV2#.magnitude
      
      
    def addFileName(self, fname):
        #adds file name attribute to object
        self.fname = fname
    def addAltitudeCharacteristics(self):
        self.lowerAlt = min(self.alt).astype('int')
        self.upperAlt = max(self.alt).astype('int')
      
    def getParameters(self):
        self.lambda_z = (self.alt[-1] - self.alt[0]) * 2
        self.m = 2 * np.pi / self.lambda_z
        #phi = eps[4]    #orientation of ellipse
        print("PHIIII", self.phi)
        rot = np.array([[np.cos(self.phi), -np.sin(self.phi)], [np.sin(self.phi), np.cos(self.phi)]])
        #print("ROTATION:", rot)
        #print("SIZE OF ROT:", rot.shape)
        uv = np.array([self.u, self.v])
        uvrot = np.matmul(rot,uv)
        urot = uvrot[0,:]
        dT = np.diff(self.temp)
        #print("Diferential Temperature:", dT)
        dz = np.diff(self.alt)
        wf = (2 * self.a) / (2 * self.b)    #long axis / short axis
        print("WFFFFFFFFFFFF", wf)
        bvMean = np.mean(self.bv2)
        coriolisFreq = mpcalc.coriolis_parameter(latitudeOfAnalysis)
        #print("Coriolis Parameter:", coriolisFreq)
        intrinsicFreq = coriolisFreq.magnitude * wf     #need to assign units to output from ellipse fitting to ensure dimensional accuracy
        k_h = np.sqrt((coriolisFreq.magnitude**2 * self.m**2) / abs(bvMean) * (wf**2 - 1)) #horizontal wavenumber
        print("KHHHHHHHHHHHHHHHHHHHHHHHHH", k_h)
        intrinsicHorizPhaseSpeed = intrinsicFreq / k_h
        k_h_2 = np.sqrt((intrinsicFreq**2 - coriolisFreq.magnitude**2) * (self.m**2 / abs(bvMean)))
        int2 = intrinsicFreq / k_h_2
        dTdz = dT / dz
        eta = np.mean(dTdz / urot[0:-1])
        
        if eta < 0:
            self.phi -= np.pi
        
        self.altOfDetection = np.mean(self.alt)
        
        
        #print("m: {}, lz: {}, h: {}, bv{}".format(self.m, self.lambda_z, intrinsicHorizPhaseSpeed, bvMean))
        
        return 
    def saveMicroHodoNoIndices(self):
        #used when the entire hodograph is to be saved
        wd = os.getcwd()
        T = np.column_stack([self.alt.magnitude, self.u.magnitude, self.v.magnitude, self.temp.magnitude, self.bv2.magnitude])
        T = pd.DataFrame(T, columns = ['Alt', 'u', 'v', 'temp', 'bv2'])
        #print("T")
        #print(T)
        
        fname = '{}-{}-{}'.format(self.fname, int(self.alt[0].magnitude), int(self.alt[-1].magnitude))
        T.to_csv('{}/microHodographs/{}.csv'.format(wd, fname), index=False, float_format='%4.3f')
        
    def saveMicroHodo(self, upperIndex, lowerIndex, fname):
        wd = os.getcwd()
        T = np.column_stack([self.alt[lowerIndex:upperIndex+1].magnitude, self.u[lowerIndex:upperIndex+1].magnitude, self.v[lowerIndex:upperIndex+1].magnitude, self.temp[lowerIndex:upperIndex+1].magnitude, self.bv2[lowerIndex:upperIndex+1].magnitude])
        T = pd.DataFrame(T, columns = ['Alt', 'u', 'v', 'temp', 'bv2'])
        #print("T")
        #print(T)
        
        fname = '{}-{}-{}'.format(fname, int(self.alt[lowerIndex].magnitude), int(self.alt[upperIndex].magnitude))
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
            

        self.a = a
        self.b = b
        self.c_x = centre[0]
        self.c_y = centre[1]
        self.phi = phi
        return 

def siftThroughUV():   #u, v, Alt in argument

    #plot Altitude vs. U,V in two subplots
    fig1 = plt.figure("U, V, hodo")

    U = plt.subplot(131)
    V = plt.subplot(132)
    microscopicHodo = plt.subplot(133)
    fig1.suptitle('Alt vs u,v')
    U.plot(u, Alt, linewidth=.5)
    V.plot(v, Alt, linewidth=.5)
    fig1.tight_layout()
    
    while True:
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
        elif string == 'nextFile':
            print("Continuing to next file")
            break
    
    print("DONE W LOOP")
    
    
  
    #result.cla()  
    
#Call functions for analysis------------------------------------------

plt.close('all')
getFiles()
preprocessDataNoResample(fileToBeInspected)
#macroHodo()
#siftThroughUV()
#eps = fit_ellipse(temporary.u, temporary.v)
#print(eps)
#hodoPicker()
#hodo_list= doAnalysis(microHodoDir)
#plotBulkMicros(hodo_list, fileToBeInspected)


manualGUI()




#_________________Artifacts_____________________________________
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




# Visualize? 