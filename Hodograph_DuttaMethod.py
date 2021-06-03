# -*- coding: utf-8 -*-
"""
Methods adopted form Dutta (2017)

Manual Hodograph Analyzer
Include this script in the same folder as the input and output directories, or else specify their file path

Semantics:
Enclosed structures contained in full-flight hodographs are reffered to as "microhodographs", 
let u_o = (U_x0, U_y0, 0) be the background wind and u_1 = (u, v, w) be the perturbed velocities 

To Do:
- add/update sources for parameter calculations
- format parameter output to match wavelet python code? Maybe doesnt make sense to use json, as profile data needs to accompany wave parameters for plotting
- get rid of metpy library?
- clean up bulk hodograph plots
- add time to wave parameter list
    

Malachi Mooney-Rivkin
Last Edit: 6/2/2021
Idaho Space Grant Consortium
moon8435@vandals.uidaho.edu
"""

#dependencies
import os
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#for ellipse fitting
from math import atan2
from numpy.linalg import eig, inv, svd

#data smoothing
from scipy import signal

#metpy related dependencies - consider removing entirely
import metpy.calc as mpcalc
from metpy.units import units

#tk gui
import tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter.font import Font
from tkinter import ttk

#skimage ellipse fitting
from skimage.measure import EllipseModel


###############################BEGINING OF USER INPUT##########################

#Which functionality would you like to use?
showVisualizations = False     # Displays macroscopic hodograph for flight
siftThruHodo = False    # Use manual GUI to locate ellipse-like structures in hodograph
analyze = False   # Display list of microhodographs with overlayed fit ellipses as well as wave parameters
location = "Tolten"     #[Tolten]/[Villarica]

#variables that are specific to analysis: These might be changed regularly depending on flight location, file format, etc.
flightData = r"C:\Users\Malachi\OneDrive - University of Idaho\%SummerInternship2020\%%CHIILE_Analysis_Backups\ChilePythonEnvironment_01112021\ChileData_012721\Tolten_01282021"             #flight data directory
fileToBeInspected = 'T26_1630_12142020_MT2.txt'                                                 #specific flight profile to be searched through manually
microHodoDir = r"C:\Users\Malachi\OneDrive - University of Idaho\workingChileDirectory\Tolten\T26_all"  
#microHodoDir = r"C:\Users\Malachi\OneDrive - University of Idaho\workingChileDirectory\Tolten\T28"              #location where selections from GUI ard. This is also the location where do analysis looks for micro hodos to analysis
waveParamDir = r"C:\Users\Malachi\OneDrive - University of Idaho\workingChileDirectory"     #location where wave parameter files are to be saved

if location == "Tolten":
    latitudeOfAnalysis = abs(-39.236248) * units.degree    #latitude at which sonde was launched. Used to account for affect of coriolis force.
elif location == "Villarica":
    latitudeOfAnalysis = abs(-39.30697) * units.degree     #same, but for Villarica...

g = 9.8                     #m * s^-2
spatialResolution = 5
heightSamplingFreq = 1/spatialResolution      #1/m used in interpolating data height-wise
minAlt = 1000 * units.m     #minimun altitude of analysis
p_0 = 1000 * units.hPa      #needed for potential temp calculatiion
movingAveWindow = 11        #need to inquire about window size selection
n_trials = 1000         #number of bootstrap iterations
#for butterworth filter
lowcut = 1500  #m - lower vertical wavelength cutoff for Butterworth bandpass filter
highcut = 4000  #m - upper vertical wavelength cutoff for Butterworth bandpass filter
order = 3   #Butterworth filter order - Dutta(2017)
##################################END OF USER INPUT######################

def preprocessDataResample(file, path, spatialResolution, lambda1, lambda2, order):
    #delete gloabal data variable as soon as troubleshooting is complete
    global data
    """ prepare data for hodograph analysis. non numeric values & values > 999999 removed, brunt-viasala freq
        calculated, background wind removed

        Different background removal techniques used: rolling average, savitsky-golay filter, nth order polynomial fits
    """
 
    #indicate which file is in progress
    print("Analyzing: {}".format(file))
    
    # Open file
    contents = ""
    f = open(os.path.join(path, file), 'r')
    print("\nOpening file "+file+":")
    for line in f:  # Iterate through file, line by line
        if line.rstrip() == "Profile Data:":
            contents = f.read()  # Read in rest of file, discarding header
            print("File contains GRAWMET profile data")
            break
    f.close()  # Need to close opened file


    # Read in the data and perform cleaning
    # Need to remove space so Virt. Temp reads as one column, not two
    contents = contents.replace("Virt. Temp", "Virt.Temp")
    # Break file apart into separate lines
    contents = contents.split("\n")
    contents.pop(1)  # Remove units so that we can read table
    index = -1  # Used to look for footer
    for i in range(0, len(contents)):  # Iterate through lines
        if contents[i].strip() == "Tropopauses:":
            index = i  # Record start of footer
    if index >= 0:  # Remove footer, if found
        contents = contents[:index]
    contents = "\n".join(contents)  # Reassemble string

    # format flight data in dataframe
    data = pd.read_csv(StringIO(contents), delim_whitespace=True)
    
    #turn strings into numeric data types, non numerics turned to nans
    data = data.apply(pd.to_numeric, errors='coerce') 

    # replace all numbers greater than 999999 with nans
    data = data.where(data < 999999, np.nan)    

    #truncate data at greatest alt
    data = data[0 : np.where(data['Alt']== data['Alt'].max())[0][0]+1]  
    print("Maximum Altitude: {}".format(max(data['Alt'])))

    #drop rows with nans
    data = data.dropna(subset=['Time', 'T', 'Ws', 'Wd', 'Long.', 'Lat.', 'Alt'])
    
    #remove unneeded columns
    data = data[['Time', 'Alt', 'T', 'P', 'Ws', 'Wd', 'Lat.', 'Long.']]
    
    #linearly interpolate data - such that it is spaced iniformly in space, heightwise - stolen from Keaton
    #create index of heights with 1 m spacial resolution - from minAlt to maxAlt
    heightIndex = pd.DataFrame({'Alt': np.arange(min(data['Alt']), max(data['Alt']))})
    #right merge data with index to keep all heights
    data= pd.merge(data, heightIndex, how='right', on='Alt')
    #sort data by height
    data = data.sort_values(by='Alt')
    #linear interpolate the nans
    missingDataLimit = 999  #more than 1km of data should be left as nans, will not be onsidered in analysis
    data = data.interpolate(method='linear', limit=missingDataLimit)
    #resample at height interval
    keepIndex = np.arange(0, len(data['Alt']), spatialResolution)
    data = data.iloc[keepIndex,:]
    data.reset_index(drop=True, inplace=True)
    
    #change data container name, sounds silly but useful for troubleshooting data-cleaning bugs
    global df
    df = data
    #print(df)
    #make following vars availabale outside of function - convenient for time being, but consider changing in future
    global Time 
    global Pres 
    global Temp 
    global Hu 
    global Wd 
    global Long 
    global Lat 
    global Alt 
    global potentialTemp
    global bv2
    global u, v 
    global uBackground 
    global vBackground
    global tempBackground

    #for comparing rolling ave to savitsky golay
    #global uBackgroundRolling
    #global vBackgroundRolling
    #global tempBackgroundRolling
    #global uRolling
    #global vRolling
    #global tRolling
    
    #individual series for each variable, local
    Time = df['Time'].to_numpy()
    Pres = df['P'].to_numpy() * units.hPa
    Temp = df['T'].to_numpy()  * units.degC
    Ws = df['Ws'].to_numpy() * units.m / units.second
    Wd = df['Wd'].to_numpy() * units.degree
    Long = df['Long.'].to_numpy()
    Lat = df['Lat.'].to_numpy()
    Alt = df['Alt'].to_numpy().astype(int) * units.meter
    
    
    #calculate brunt-viasala frequency **2 
    tempK = Temp.to('kelvin')
    
    potentialTemperature =  tempK * (p_0 / Pres) ** (2/7)    #https://glossary.ametsoc.org/wiki/Potential_temperature   
    bv2 = mpcalc.brunt_vaisala_frequency_squared(Alt, potentialTemperature).magnitude    #N^2 
    #bv2 = bruntViasalaFreqSquared(potentialTemperature, heightSamplingFreq)     #Maybe consider using metpy version of N^2 ? Height sampling is not used in hodo method, why allow it to affect bv ?
    
    #convert wind from polar to cartesian c.s.
    u, v = mpcalc.wind_components(Ws, Wd)   #raw u,v components - no different than using trig fuctions
    print("Size of u: ", len(u))
    #subtract nth order polynomials to find purturbation profile
    
    # run moving average over u,v comps
    altExtent = max(Alt) - minAlt    #NEED TO VERIFY THE CORRECT WINDOW SAMPLING SZE
    print("Alt Extent:", altExtent)
    #window = int((altExtent.magnitude / (heightSamplingFreq * 4)))    # as done in Tom's code; arbitrary at best. removed choosing max between calculated window and 11,artifact from IDL code
    #if (window % 2) == 0:       #many filters require odd window
    #    window = window-1
        

    #################de-trend that matches keatons code############
    # Subtract rolling mean (assumed to be background wind)
    #N = int(len(Alt) / 4)
    #print("Window Size N: ", N)
    # Also, figure out what min_periods is really doing and make a reason for picking a good value
    #uBackgroundRolling = pd.Series(u.magnitude).rolling(window=N, min_periods=int(N/2), center=True).mean().to_numpy() * units.m/units.second
    #uRolling = u - (uBackgroundRolling.to_numpy() * units.m/units.second)
    #vBackgroundRolling = pd.Series(v.magnitude).rolling(window=N, min_periods=int(N/2), center=True).mean().to_numpy() * units.m/units.second
    #vRolling = v - (vBackgroundRolling.to_numpy() * units.m/units.second)
    #tempBackgroundRolling = pd.Series(Temp.magnitude).rolling(window=N, min_periods=int(N / 2), center=True).mean().to_numpy() * units.degC
    #tRolling = Temp - (tempBackgroundRolling.to_numpy() * units.degC)
    ###################end de-trend that matches keatons code###########
    
    #de-trend u, v, temp series; NEED TO RESEARCH MORE, rolling average vs. fft vs. polynomial fit vs. others?
    #uBackground = signal.savgol_filter(u.magnitude, window, 3, mode='mirror') * units.m/units.second        #savitsky-golay filter fits polynomial to moving window
    #vBackground = signal.savgol_filter(v.magnitude, window, 3, mode='mirror') * units.m/units.second
    #tempBackground = signal.savgol_filter(Temp.magnitude, window, 3, mode='mirror') * units.degC
    
    #detrend  temperature using polynomial fit
    #Fig = plt.figure(1)
    Fig, axs = plt.subplots(2,4,figsize=(6,6), num=1)   #figure for temperature
    Fig2, axs2 = plt.subplots(2,4,figsize=(6,6), num=2)   #figure for wind
    
    axs = axs.flatten() #make subplots iteratble by single indice
    axs2 = axs2.flatten()
    temp_background = []
    u_background = []
    v_background = []
    
    for k in range(2,10):
        i = k-2
        
        #temp
        poly = np.polyfit(Alt.magnitude / 1000, Temp.magnitude, k)
        fit = np.polyval(poly, Alt.magnitude / 1000)
        temp_background.append(fit)
        
        #plot
        axs[i].plot(fit, Alt.magnitude / 1000, color='darkblue')
        axs[i].plot(Temp.magnitude, Alt.magnitude / 1000)
        axs[i].set_title("Order: " + str(k))
        #axs[i].set_xlabel("Temperature (C)")
        axs[i].set_ylabel("Altitude (km)")
        axs[i].tick_params(top=True, right=True)
        
        #u
        poly = np.polyfit(Alt.magnitude / 1000, u.magnitude, k)
        fit = np.polyval(poly, Alt.magnitude / 1000)
        u_background.append(fit)
        #plot u
        zonal, = axs2[i].plot(fit, Alt.magnitude / 1000, color='darkblue', label='Zonal')
        axs2[i].plot(u.magnitude, Alt.magnitude / 1000, color='darkblue')
        axs2[i].set_title("Order: " + str(k))
        #axs[i].set_xlabel("Temperature (C)")
        axs2[i].set_ylabel("Altitude (km)")
        axs2[i].tick_params(top=True, right=True)
        
        #v
        poly = np.polyfit(Alt.magnitude / 1000, v.magnitude, k)
        fit = np.polyval(poly, Alt.magnitude / 1000)
        v_background.append(fit)
        #plot v
        meridional, = axs2[i].plot(fit, Alt.magnitude / 1000, color='darkred', label='Meridional')
        axs2[i].plot(v.magnitude, Alt.magnitude / 1000, color='darkred')
        #axs2[i].set_title("Order: " + str(k))
        #axs[i].set_xlabel("Temperature (C)")
        #axs2[i].set_ylabel("Altitude (km)")
        #axs2[i].tick_params(top=True, right=True)
        
        
        
    #Fig - hide labels and ticks, add notations
    plt.figure(1)
    Fig.add_subplot(111, frameon=False) #make hidden subplot to add xlabel
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False) #which='both'
    plt.xlabel("Temperature (C)")
    Fig.suptitle("Temperature; Polynomial Fitted \n {}".format(file))
    for ax in axs:
        ax.label_outer()
    
    #Fig2 - hide labels and ticks, add notations
    plt.figure(2)
    Fig2.suptitle("Wind; Polynomial Fitted \n {}".format(file))
    for ax in axs2:
        ax.label_outer()
    Fig2.add_subplot(111, frameon=False) #make hidden subplot to add xlabel
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False) #which='both'
    plt.xlabel("Winds (m/s)")
    #handles, labels = axs2.get_legend_handles_labels()
    #Fig2.legend(handles,labels, loc='lower center')
    Fig2.legend(handles=[zonal, meridional], labels=['Zonal', 'Meridional'], loc='lower center', ncol=2)
    
    #subtract background
    #u -= uBackgroundRolling 
    #v -= vBackgroundRolling 
    #Temp -= tempBackgroundRolling
    
    #subtract fits to produce various perturbation profiles
    tempPert = []
    global uPert
    uPert = []
    vPert = []
    
    for i, element in enumerate(temp_background):
        pert = np.subtract(Temp.magnitude, temp_background[i])
        tempPert.append(pert)
        pert = np.subtract(u.magnitude, u_background[i])
        uPert.append(pert)
        pert = np.subtract(v.magnitude, v_background[i])
        vPert.append(pert)
        
    
    #plot to double check subtraction
    Fig, axs = plt.subplots(2,2,figsize=(6,6), num=3, sharey=True)#, sharex=True)   #figure for u,v butterworth filter
    for i,element in enumerate(u_background):
        axs[0,0].plot(uPert[i], Alt.magnitude/1000, linewidth=0.5, label="Order: {}".format(str(i+2)))
        axs[1,0].plot(vPert[i], Alt.magnitude/1000, linewidth=0.5)   
   
    Fig.legend()
    Fig.suptitle("Wind Components; Background Removed, Filtered \n {}".format(file))
    axs[0,0].set_xlabel("Zonal Wind (m/s)")
    axs[1,0].set_xlabel("Meridional Wind (m/s)")
    axs[0,1].set_xlabel("Filtered Zonal Wind (m/s)")
    axs[0,1].set_xlim([-10,10])
    axs[1,1].set_xlim([-10,10])
    axs[0,0].set_xlim([-20,35])
    axs[1,0].set_xlim([-20,35])
    axs[1,1].set_xlabel("Filtered Meridional Wind (m/s)")
    axs[0,0].set_ylabel("Altitude (km)")
    axs[1,0].set_ylabel("Altitude (km)")
    axs[0,0].tick_params(axis='x',labelbottom=False) # labels along the bottom edge are off
    axs[0,1].tick_params(axis='x',labelbottom=False) # labels along the bottom edge are off
    ###############################################################
    ###############################################################
    
    #filter using 3rd order butterworth - fs=samplerate (1/m)
    freq2 = 1/lambda1    #find cutoff freq 1/m
    freq1 =  1/lambda2    #find cutoff freq 1/m
    
    # Plot the frequency response for a few different orders.
    b, a = butter_bandpass(freq1, freq2, heightSamplingFreq, order)
    w, h = signal.freqz(b, a, worN=5000)
    plt.figure(4)
    plt.plot(w/np.pi, abs(h))
    plt.plot([0, 1], [np.sqrt(0.5), np.sqrt(0.5)],'--', label='sqrt(1/2)')
    plt.xlabel('Normalized Frequency (x Pi rad/sample) \n [Nyquist Frequency = 1]') #1/m ?
    plt.ylabel('Gain')
    plt.xlim([0,.1])
    plt.grid(True)
    plt.title("Frequency Response of 3rd Order Butterworth Filter \n Vertical Cut-off Wavelengths: 1.5 - 4 km")
    plt.legend(loc='best')

    # Filter a noisy signal.
    uButter = []
    vButter = []
    
    for i,element in enumerate(vPert):
        
        filtU = butter_bandpass_filter(uPert[i],freq1, freq2, heightSamplingFreq, order)
        uButter.append(filtU)
        filtV = butter_bandpass_filter(vPert[i], freq1, freq2, 1/5, order)
        vButter.append(filtV)
        #axs[1,1].plot(vPert[0], Alt.magnitude)
        axs[1,1].plot(vButter[i], Alt.magnitude/1000, linewidth=0.5)
        axs[0,1].plot(uButter[i], Alt.magnitude/1000, linewidth=0.5)
        #plt.xlabel('time (seconds)')
        #plt.hlines([-a, a], 0, T, linestyles='--')
        #plt.grid(True)
        #plt.axis('tight')
        #plt.legend(loc='upper left')
    
    #######
    ###########
    #########

def butter_bandpass(lowcut, highcut, fs, order):
    """
        Used for plotting the frequency response of Butterworth
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    """
        Applies Butterworth filter to perturbation profiles
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y



  

def bruntViasalaFreqSquared(potTemp, heightSamplingFreq):
    """ replicated from Tom's script
    """
    G = 9.8 * units.m / units.second**2
    N2 = (G / potTemp) * np.gradient(potTemp, heightSamplingFreq * units.m)     #artifact of tom's code, 
    return N2

class microHodo:
    def __init__(self, ALT, U, V, TEMP, BV2, LAT, LONG, TIME):
      self.alt = ALT#.magnitude
      self.u = U#.magnitude
      self.v = V#.magnitude
      self.temp = TEMP#.magnitude
      self.bv2 = BV2#.magnitude
      self.lat = LAT
      self.long = LONG
      self.time = TIME
      
      
    def addOrientation(self, ORIENTATION):
        self.orientation = np.full((len(self.time), 1), ORIENTATION)
      
      
    def addNameAddPath(self, fname, fpath):
        #adds file name attribute to object
        self.fname = fname
        self.savepath = fpath
        
    def addAltitudeCharacteristics(self):
        self.lowerAlt = min(self.alt).astype('int')
        self.upperAlt = max(self.alt).astype('int')
      
    def getParameters(self):

        #Altitude of detection - mean
        self.altOfDetection = np.mean(self.alt)     # (meters)

        #Latitude of Detection - mean
        self.latOfDetection = np.mean(self.lat)     # (decimal degrees) 

        #Longitude of Detection - mean
        self.longOfDetection = np.mean(self.long)     # (decimal degrees)

        #Date/Time of Detection - mean - needs to be added!
        
        #Axial ratio
        wf = (2 * self.a) / (2 * self.b)    #long axis / short axis
        

        #Vertical wavelength
        self.lambda_z = self.alt[-1] - self.alt[0]       # (meters) -- Toms script multiplies altitude of envelope by two? 
        self.m = 2 * np.pi / self.lambda_z      # vertical wavenumber (rad/meters)

        #Horizontal wavelength
        bv2Mean = np.mean(self.bv2)
        coriolisFreq = mpcalc.coriolis_parameter(latitudeOfAnalysis)
        
        k_h = np.sqrt((coriolisFreq.magnitude**2 * self.m**2) / abs(bv2Mean) * (wf**2 - 1)) #horizontal wavenumber (1/meter)
        self.lambda_h = 1 / k_h     #horizontal wavelength (meter)

        #Propogation Direction (Marlton 2016) 
        
        #rot = np.array([[np.cos(self.phi), -np.sin(self.phi)], [np.sin(self.phi), np.cos(self.phi)]])       #2d rotation matrix - containinng angle of fitted elipse - as used in Toms script
        rot = np.array([[np.cos(-self.phi), -np.sin(-self.phi)], [np.sin(-self.phi), np.cos(-self.phi)]])       #2d rotation matrix - containinng  negative angle of fitted elipse
        uv = np.array([self.u, self.v])       #zonal and meridional components
        uvrot = np.matmul(rot,uv)       #change of coordinates
        urot = uvrot[0,:]               #urot aligns with major axis
        #print('UROT MAX', max(urot))
        dt = np.diff(self.temp)
        #print("dt: ", dt)
        dz = np.diff(self.alt)
        #print('dz: ', dz)
        dTdz = np.diff(self.temp)  / np.diff(self.alt) 
        #print('dTdz: ', dTdz)             #discreet temperature gradient dt/dz

        ###EXPERIMENT TO CHECK HOW HEIGHT CHANGES WITH TEMP
        dzdT = np.diff(self.alt)  / np.diff(self.temp) 

        ###END EXPERIMENT#################################
        eta = np.mean(dTdz / urot[0:-1])
        if eta < 0:                 # check to see if temp perterbaton has same sign as u perterbation - clears up 180 deg ambiguity in propogation direction
            self.phi += np.pi
        
        self.directionOfPropogation = self.phi      # (radians ccw fromxaxis)
        self.directionOfPropogation = np.rad2deg(self.directionOfPropogation)
        #self.directionOfPropogation = 450 - self.directionOfPropogation
        if self.directionOfPropogation > 360:
            self.directionOfPropogation -= 360

        """
        ########################################PLOTTING FOR TROUBLESHOOTING##################################################
        #plots all micro-hodographs for a single flight
        uvPlot = plt.figure("Troubleshooting", figsize=(10, 5))
        
        ax = uvPlot.add_subplot(1,2,1, aspect='equal')
        ax.plot(self.u, self.v, 'red') 

        #plot parametric best fit ellipse
        param = np.linspace(0, 2 * np.pi)
        x = self.a * np.cos(param) * np.cos(self.phi) - self.b * np.sin(param) * np.sin(self.phi) + self.c_x
        y = self.a * np.cos(param) * np.sin(self.phi) + self.b * np.sin(param) * np.cos(self.phi) + self.c_y
        ax.plot(x, y)
        ax.set_xlabel("(m/s)")
        ax.set_ylabel("(m/s)")
        ax.set_aspect('equal')
        ax.set_title("UV with fit ellipse")
        ax.grid()
        #plot uvRot
        ax.plot(urot, uvrot[1,:])
        

        #plot u, t, vs alt
        color = 'tab:red'
        ax = uvPlot.add_subplot(1,2,2, )
        ax.set_xlabel('urot (m/s)', color=color)
        ax.set_ylabel('alt (m)')
        ax.plot(urot, self.alt, label='urot', color=color)
        ax.plot(self.temp, self.alt, label='temp', color = 'green')
        ax.tick_params(axis='x', labelcolor=color)

        ax2 = ax.twiny()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_xlabel('dtdz', color=color)  # we already handled the y-label with ax1
        ax2.plot(dTdz, self.alt[0:-1], color=color, label='dTdz')
        ax2.tick_params(axis='x', labelcolor=color)

        ax.legend()  
            
        plt.show() 


        #########################################END PLOTTING FOR TROUBLESHOOTING#############################################
        """
        #Intrinsic vertical group velocity

        #Intrinsic horizontal group velocity

        #Intrinsic vertical phase speed
        

        #Intrinsic horizontal phase speed (m/s)
        intrinsicFreq = coriolisFreq.magnitude * wf     #one ought to assign units to output from ellipse fitting to ensure dimensional accuracy
        intrinsicHorizPhaseSpeed = intrinsicFreq / k_h

        #extraneous calculations - part of Tom's script
        #k_h_2 = np.sqrt((intrinsicFreq**2 - coriolisFreq.magnitude**2) * (self.m**2 / abs(bv2Mean)))
        #int2 = intrinsicFreq / k_h_2

        #print("m: {}, lz: {}, h: {}, bv{}".format(self.m, self.lambda_z, intrinsicHorizPhaseSpeed, bv2Mean))
        #return altitude of detection, latitude, longitude, vertical wavelength,horizontal wavenumber, intrinsic horizontal phase speed, axial ratio l/s
        return  [self.time, self.altOfDetection, self.lat[0], self.long[0], self.lambda_z, k_h, intrinsicHorizPhaseSpeed, wf, self.directionOfPropogation]

    def saveMicroHodoNoIndices(self):
        """ dumps microhodograph object attributs into csv 
        """
    
        T = np.column_stack([self.time, self.alt.magnitude, self.u.magnitude, self.v.magnitude, self.temp.magnitude, self.bv2, self.lat, self.long, self.orientation]) 
        T = pd.DataFrame(T, columns = ['time', 'alt', 'u', 'v', 'temp', 'bv2', 'lat','long', 'orientation'])
        
        
        
        fname = '{}_microHodograph_{}-{}'.format(self.fname.strip('.txt'), int(self.alt[0].magnitude), int(self.alt[-1].magnitude))
        T.to_csv('{}/{}.csv'.format(self.savepath, fname), index=False)                          

    #ellipse fitting courtesy of  Nicky van Foreest https://github.com/ndvanforeest/fit_ellipse
    # a least squares algorithm is used
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
        return a, b, centre[0], centre[1],phi
    
    def bootstrap_params(self, n_trials):
        uu, vv = np.asarray(self.u), np.asarray(self.v)
        #arrays to store parameters of all fits
        aas, bs, c_xs, c_ys, phis = np.zeros(n_trials), np.zeros(n_trials), np.zeros(n_trials), np.zeros(n_trials), np.zeros(n_trials)
        
        a, b, c_x, c_y,phi = self.fit_ellipse() # first trial
        aas[0] = a
        bs[0] = b
        c_xs[0] = c_x
        c_ys[0] = c_y
        phis[0] = phi
        
 
        num_data_points = self.u.shape[0] # or len(u), depending on if it's an array or list
        
        #keep track of erroneous fits
        badIndices = []
        for i in range(1, n_trials):
 
            random_indices = np.random.choice(num_data_points, size=num_data_points,replace=True) # grab indices with repetition to subsample your data
            random_subset_u = uu[random_indices] # make sure u and v are numpy arrays
            random_subset_v = vv[random_indices] # with u = np.asarray(u) if u is a list
            self.u = random_subset_u
            self.v = random_subset_v
            #print("random indices: ", random_indices)                               
            a, b, c_x, c_y,phi = self.fit_ellipse() # get estimate of params for this bootstrapped sample
                
                
            if np.isnan(a) | np.isnan(b): 
                badIndices.append(i)
            else:                           
                aas[i] = a
                bs[i] = b
                c_xs[i] = c_x
                c_ys[i] = c_y
                phis[i] = phi
         
            #return u,v to unperturbed state
            self.u = uu
            self.v = vv
        
            
        #get rid of iterations with bad indices
        print("bad indices: ", len(badIndices))
        aas = np.delete(aas, badIndices)
        bs = np.delete(bs, badIndices)
        c_xs = np.delete(c_xs, badIndices)
        c_ys = np.delete(c_ys, badIndices)
        phis = np.delete(phis, badIndices)
        
        sdAR = np.std(aas/bs)
        
        #create bootstrap values-means
        self.bsa = np.nanmean(aas)
        self.bsb = np.nanmean(bs)
        self.bsc_x = np.nanmean(c_xs)
        self.bsc_y = np.nanmean(c_ys)
        self.bsphi = np.nanmean(phis)
        
        
        print("a,b,x,y,phi: ", self.bsa,self.bsb,self.bsc_x,self.bsc_y,self.bsphi)
        print("STDev AR: ", sdAR)
        
        return np.mean(aas), np.std(aas), np.mean(bs), np.std(bs), np.mean(c_xs), np.std(c_xs), np.mean(c_ys), np.std(c_ys), np.mean(phis), np.std(phis)# return statistics
"""
    def improveFit(self):
        uu, vv = np.asarray(self.u), np.asarray(self.v)
        #get initial estimate of parameters
        a, b, c_x_old, c_y_old,phi = self.fit_ellipse() # get estimate of params
        
        #estimate center
        c_x_temp = np.mean(self.u)
        c_y_temp = np.mean(self.v)
        
        #subtract mean - this centers hodograph at origin
        self.u -= c_x_old
        self.v -= c_y_old
        
        #rotate ellipse to align major axis with u
        rot = np.array([[np.cos(-self.phi), -np.sin(-self.phi)], [np.sin(-self.phi), np.cos(-self.phi)]])       #2d rotation matrix - containinng  negative angle of fitted elipse
        uv = np.array([self.u, self.v])       #zonal and meridional components
        uvrot = np.matmul(rot,uv)       #change of coordinates
        self.urot = uvrot[0,:]               #urot aligns with major axis
        self.vrot= uvrot[1,:]
        self.u = uvrot[0,:]               #urot aligns with major axis
        self.v= uvrot[1,:]
        
        
        
        #scale minor axis by std of major axis this is fairly abitrary
        stdev_x = np.std(self.u)
        stdev_y = np.std(self.v)
        ratioOfEccentricity = stdev_x/stdev_y
        #ratioOfEccentricity = a/b
        self.v = self.v * ratioOfEccentricity
        
        #keep track of progression...
        self.uMod = self.u
        self.vMod = self.v
        
        #fit ellipse to rotated data
        a, b, c_x_new, c_y_new, phi = self.fit_ellipse() # get estimate of params
        self.arot = a
        self.brot = b
        self.c_x_improved = c_x_new+c_x_old
        self.c_y_improved = c_y_new+c_y_old
        self.phirot = phi
        self.c_x_rot = c_x_new
        self.c_y_rot = c_y_new
        
        
        #scale parameters back
        self.b_improved = b / ratioOfEccentricity
        self.a_improved = a
        
        
        
        
        
        
        #return u,v t original state
        self.u = uu
        self.v = vv
        a, b, c_x, c_y,phi = self.fit_ellipse() # get estimate of params
           
"""
def doAnalysis(microHodoDir):
    """ Extracts wave parameters from microHodographs; this function can be run on existing microhodograph files without needing to operate the GUI
    """
    #make sure files are retrieved from correct directory; consider adding additional checks to make sure user is querying correct directory
    print("Micro Hodograph Path Exists: ", os.path.exists(microHodoDir))
    
    hodo_list = []
    parameterList = []
    print("all files in path:", os.listdir(microHodoDir))
    for file in os.listdir(microHodoDir):
        path = os.path.join(microHodoDir, file)
        print('Analyzing micro-hodos for flight: {}'.format(file))
        
        #dataframe from local hodograph file
        df = np.genfromtxt(fname=path, delimiter=',', names=True)
    
        #create microhodograph object, then start giving it attributes
        instance = microHodo(df['Alt'], df['u'], df['v'], df['temp'], df['bv2'], df['lat'], df['long'], df['time'])

        #file name added to object attribute here to be used in labeling plots
        instance.addNameAddPath(file, microHodoDir)  

        #find out min/max altitudes file
        instance.addAltitudeCharacteristics()
        
        #test an improved ellipse fit
        #instance.improveFit()
        
        #lets try to fit an ellipse to microhodograph
        instance.bootstrap_params(n_trials)
        instance.fit_ellipse()
        
        
        
        #use ellipse to extract wave characteristics
        params = instance.getParameters()
        #print("Wave Parameters: \n", params)

        #update running list of processed hodos and corresponding parameters
        parameterList.append(params)
        hodo_list.append(instance)  #add micro-hodo to running list
    
    #organize parameters into dataframe; dump into csv
    parameterList = pd.DataFrame(parameterList, columns = ['time', 'Alt.', 'Lat', 'Long', 'Vert Wavelength', 'Horizontal Wave#', 'IntHorizPhase Speed', 'Axial Ratio L/S', 'Propagation Direction' ])
    parameterList.sort_values(by='Alt.', inplace=True)
    
    pathAndFile = "{}\{}_params.csv".format(waveParamDir, fileToBeInspected.strip(".txt"))
    parameterList.to_csv(pathAndFile, index=False, na_rep='NaN')
    
    #sort list of hodographs in order of ascending altitude 
    hodo_list.sort(key=lambda x: x.altOfDetection)  

    return hodo_list     
    
def plotBulkMicros(hodo_list, fname):
    """ plot microhodographs in grid of subplots
    """ 
    
    #plots all micro-hodographs for a single flight
    bulkPlot = plt.figure(fname, figsize=(8.5,11))
    plt.suptitle("Micro-hodographs for \n {}".format(fname))#, y=1.09)
    
    totalPlots = len(hodo_list)
    if totalPlots > 20:
        bulkplot2 = plt.figure(fname+" pt2", figsize=(8.5,11))
    
    if totalPlots > 0:
        
        #figure out how to arrang subplots on grid
        numColumns = 4
        numRows = 5
        position = range(1, totalPlots + 1)
        
        
        i = 0   #counter for indexing micro-hodo objects
        for hodo in hodo_list:
            print("HODO ITERATION: ", hodo)
            #ax = bulkPlot.add_subplot(numRows, numColumns, position[i])#, aspect='equal')
            fig, ax = plt.subplots(figsize=(8.5,8.5))
            ax.plot(hodo_list[i].u, hodo_list[i].v, color='black', label='hodograph') 
            #axs[i].plot(hodo_list[i].u, hodo_list[i].v)
            
            #plot rotated raw data
            #ax.plot(hodo_list[i].urot, hodo_list[i].vrot, color='black', label='rotated', linewidth=1) 
            #plot scaled raw data
            #ax.plot(hodo_list[i].uMod, hodo_list[i].vMod, color='blue', label='scaled', linewidth=1)
        
            #plot parametric best fit ellipse
            param = np.linspace(0, 2 * np.pi)
            #best fit
            x = hodo_list[i].a * np.cos(param) * np.cos(hodo_list[i].phi) - hodo_list[i].b * np.sin(param) * np.sin(hodo_list[i].phi) + hodo_list[i].c_x
            y = hodo_list[i].a * np.cos(param) * np.sin(hodo_list[i].phi) + hodo_list[i].b * np.sin(param) * np.cos(hodo_list[i].phi) + hodo_list[i].c_y
            ax.plot(x, y, color='red', label='single fit')
            #axs[i].plot(x, y, color='red', label='single fit')
            
            #rotated fit
            #x = hodo_list[i].arot * np.cos(param) * np.cos(hodo_list[i].phirot) - hodo_list[i].brot * np.sin(param) * np.sin(hodo_list[i].phirot) + hodo_list[i].c_x_rot
            #y = hodo_list[i].arot * np.cos(param) * np.sin(hodo_list[i].phirot) + hodo_list[i].brot * np.sin(param) * np.cos(hodo_list[i].phirot) + hodo_list[i].c_y_rot
            #ax.plot(x, y, color='blue', label='rotated fit')
            
            #improved fit
            #x = hodo_list[i].a_improved * np.cos(param) * np.cos(hodo_list[i].phi) - hodo_list[i].b_improved * np.sin(param) * np.sin(hodo_list[i].phi) + hodo_list[i].c_x_improved
            #y = hodo_list[i].a_improved * np.cos(param) * np.sin(hodo_list[i].phi) + hodo_list[i].b_improved * np.sin(param) * np.cos(hodo_list[i].phi) + hodo_list[i].c_y_improved
            #ax.plot(x, y, color='purple', label='improved fit')
            #axs[i].plot(x, y, color='red', label='single fit')
            
            #try using skimage to fit
            xs = hodo_list[i].u
            ys = hodo_list[i].v
            a_points = np.array([xs, ys])
            print("SHAPE OF SKIMAGE ARRAY: ", np.shape(a_points.transpose()))
            ell = EllipseModel()
            ell.estimate(a_points.transpose())
            xc_skimage, yc_skimage, a_skimage, b_skimage, theta_skimage = ell.params
            
            #improved fit
            x = a_skimage * np.cos(param) * np.cos(theta_skimage) - b_skimage * np.sin(param) * np.sin(theta_skimage) + xc_skimage
            y = a_skimage * np.cos(param) * np.sin(theta_skimage) + b_skimage * np.sin(param) * np.cos(theta_skimage) + yc_skimage
            ax.plot(x, y, color='orange', label='improved fit')
            
            
            #bootstrapped values
            #x = hodo_list[i].bsa * np.cos(param) * np.cos(hodo_list[i].bsphi) - hodo_list[i].bsb * np.sin(param) * np.sin(hodo_list[i].bsphi) + hodo_list[i].bsc_x
            #y = hodo_list[i].bsa * np.cos(param) * np.sin(hodo_list[i].bsphi) + hodo_list[i].bsb * np.sin(param) * np.cos(hodo_list[i].bsphi) + hodo_list[i].bsc_y
            #ax.plot(x, y, color='blue', label='bootstrapped fit')
            
            ax.set_xlabel("(m/s)", fontsize=7)
            ax.set_ylabel("(m/s)", fontsize=7)
            ax.set_aspect('equal')
            ax.set_title("{}-{} (m)".format(hodo_list[i].lowerAlt, hodo_list[i].upperAlt), fontsize=9 )
            #if i==0:
            ax.legend(prop={'size':9})
            
            i += 1
            #plt.xlim([-5, 5])
            #plt.ylim([-5,5])
        
        plt.subplots_adjust(top=0.9, hspace=.6) 
        #plt.tight_layout(pad=1, w_pad=.5, h_pad=.5)   
        plt.tight_layout()
        
        plt.show() 
        return

def macroHodo():
    """ plot hodograph for entire flight
    """
    #plot v vs. u
    plt.figure("Macroscopic Hodograph", figsize=(10,10))  #Plot macroscopic hodograph
    plt.suptitle("Macro Hodograph for Entire Flight \n Background Wind Removed")
    c=Alt
    plt.scatter( u, v, c=c, cmap = 'magma', s = 1, edgecolors=None, alpha=1)
    #plt.plot(u,v)
    cbar = plt.colorbar()
    cbar.set_label('Altitude')  
    return

'''Interesting book recomendation: The tipping point by Malcolm Gladwell // Michael Pollan [coffee?]; per Dave Brown's reccommendation...'''

def uvVisualize():
    """ show u, v, background wind vs. altitude
    """
    #housekeeping
    plt.figure("U & V vs Time", figsize=(10,10)) 
    plt.suptitle('Smoothed U & V Components', fontsize=16)

    #u vs alt
    plt.subplot(1,2,1)
    plt.plot((u.magnitude + uBackground.magnitude), Alt.magnitude, label='Raw')
    plt.plot(uBackground.magnitude, Alt.magnitude, label='Background - S.G.')
    plt.plot(u.magnitude, Alt.magnitude, label='De-Trended - S.G.')

    #plt.plot(uBackgroundRolling, Alt.magnitude, label='Background - R.M.')
    #plt.plot(uRolling, Alt.magnitude, label='De-Trended - R.M.')

    plt.xlabel('(m/s)', fontsize=12)
    plt.ylabel('(m)', fontsize=12)
    plt.title("U")

    #v vs alt
    plt.subplot(1,2,2)
    plt.plot((v.magnitude + vBackground.magnitude), Alt.magnitude, label='Raw')
    plt.plot(vBackground.magnitude, Alt.magnitude, label='Background - S.G.')
    plt.plot(v.magnitude, Alt.magnitude, label='De-Trended - S.G.')

    #plt.plot(vBackgroundRolling, Alt.magnitude, label='Background - R.M.')
    #plt.plot(vRolling, Alt.magnitude, label='De-Trended - R.M.')

    plt.xlabel('(m/s)', fontsize=12)
    plt.ylabel('(m)', fontsize=12)
    plt.legend(loc='upper right', fontsize=16)
    plt.title("V")
    return



def manualTKGUI():
    """ tool for visualizing hodograph, sifting for micro hodographs, saving files for future analysis
    """
    
    class App:
        def __init__(self, master):
            """ method sets up gui
            """
            
            alt0 = 0
            wind0 = 100
            # Create a container
            tkinter.Frame(master)
            
            
            
            #CREATE ORIENTATION SELECTION
            
            self.orient = StringVar()
            self.orient.set(['CW', 'CCW'])
            self.picDir = tkinter.Spinbox(root, textvariable=self.orient, values=['CW', 'CCW'], state='readonly', wrap=True, font=Font(family='Helvetica', size=18, weight='normal'))
            self.picDir.place(relx=.05, rely=.32, relheight=.05, relwidth=.15)
            
            #END ORIENTATION SELECTION
            
            
            #Create Sliders
            self.alt = IntVar()
            self.win = IntVar()
            
            

            #initialize gui vars added 12/27
            #self.alt.set(min(Alt.magnitude.tolist())) 
            
            #self.altSpinner = tkinter.Spinbox(root, command=self.update, textvariable=self.alt, values=Alt.magnitude.tolist(), font=Font(family='Helvetica', size=25, weight='normal')).place(relx=.05, rely=.12, relheight=.05, relwidth=.15)
            #added 12/27 test
            self.altSpinner = tkinter.Spinbox(root, command=self.update, values=Alt.magnitude.tolist(), repeatinterval=1, font=Font(family='Helvetica', size=25, weight='normal'))
            self.altSpinner.place(relx=.05, rely=.12, relheight=.05, relwidth=.15)  #originally followed above line
            #self.winSpinner = tkinter.Spinbox(root, command=self.update, textvariable=self.win, from_=5, to=1000, font=Font(family='Helvetica', size=25, weight='normal')).place(relx=.05, rely=.22, relheight=.05, relwidth=.15)
            self.winSpinner = tkinter.Spinbox(root, command=self.update, from_=5, to=1000, repeatinterval=1, font=Font(family='Helvetica', size=25, weight='normal'))
            self.winSpinner.place(relx=.05, rely=.22, relheight=.05, relwidth=.15)  #originally followed above line
            self.altLabel = tkinter.Label(root, text="Select Lower Altitude (m):", font=Font(family='Helvetica', size=18, weight='normal')).place(relx=.05, rely=.09)
            self.winLabel = tkinter.Label(root, text="Select Alt. Window (# data points):", font=Font(family='Helvetica', size=18, weight='normal')).place(relx=.05, rely=.19)
            
            #Create figure, plot 
            fig = Figure(figsize=(5, 4), dpi=100)
            self.ax = fig.add_subplot(111)
            fig.suptitle("{}".format(fileToBeInspected))
            self.l, = self.ax.plot(u[:alt0+wind0], v[:alt0+wind0], 'o', ls='-', markevery=[0])
            self.ax.set_aspect('equal')
        
            self.canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
            self.canvas.draw()
            self.canvas.get_tk_widget().place(relx=0.25, rely=0.05, relheight=.9, relwidth=.7)
            #frame.pack()
            
            self.winLabel = tkinter.Label(root, text="Blue dot indicates lower altitude", font=Font(family='Helvetica', size=15, weight='normal')).place(relx=.05, rely=.4)
            self.quitButton = tkinter.Button(master=root, text="Quit", command=self._quit).place(relx=.05, rely=.6, relheight=.05, relwidth=.15)
            self.saveButton = tkinter.Button(master=root, text="Save Micro-Hodograph", command=self.save).place(relx=.05, rely=.5, relheight=.05, relwidth=.15)

            self.readyToSave = False #flag to make sure hodo is updated before saving
            #---------
            
        def update(self, *args):
            """ on each change to gui, this method refreshes hodograph plot
            """
            self.readyToSave = True
            #sliderAlt = int(self.alt.get()) works originally
            sliderAlt = int(float(self.altSpinner.get()))
            sliderWindow = int(self.winSpinner.get())
            self.l.set_xdata(u[np.where(Alt.magnitude == sliderAlt)[0][0]:np.where(Alt.magnitude == sliderAlt)[0][0] + sliderWindow])
            self.l.set_ydata(v[np.where(Alt.magnitude == sliderAlt)[0][0]:np.where(Alt.magnitude == sliderAlt)[0][0] + sliderWindow])
           
            self.ax.autoscale(enable=True)
            self.ax.relim()
            self.canvas.draw()
            return
        
        def save(self): 
            """ save current visible hodograph to .csv for further analysis
            """
            if self.readyToSave:
                
                ORIENTATION = self.orient.get()
                print('Orientation: ', ORIENTATION )
                sliderAlt = int(float(self.altSpinner.get()))
                sliderWindow = int(float(self.winSpinner.get()))
                lowerAltInd = np.where(Alt.magnitude == sliderAlt)[0][0]
                upperAltInd = lowerAltInd + sliderWindow
            
            
                #collect local data for altitude that is visible in plot, dump into .csv
                ALT = Alt[lowerAltInd : upperAltInd]
                U = u[lowerAltInd : upperAltInd]
                V = v[lowerAltInd : upperAltInd]
                TEMP = Temp[lowerAltInd : upperAltInd]
                BV2 = bv2[lowerAltInd : upperAltInd]
                LAT = Lat[lowerAltInd : upperAltInd]
                LONG = Long[lowerAltInd : upperAltInd]
                TIME = Time[lowerAltInd : upperAltInd]
                
                instance = microHodo(ALT, U, V, TEMP, BV2, LAT, LONG, TIME)
                instance.addOrientation(ORIENTATION)
                instance.addNameAddPath(fileToBeInspected, microHodoDir)
                instance.saveMicroHodoNoIndices()
                
            else: 
                print("Please Update Hodograph")
                    
            
            return

        def _quit(self):
            """ terminate gui
            """
            root.quit()     # stops mainloop
            root.destroy()  # this is necessary on Windows to prevent Fatal Python Error: PyEval_RestoreThread: NULL tstate
    
    root = tkinter.Tk()
    root.geometry("400x650")
    root.wm_title("Manual Hodograph GUI")
    App(root)
    root.mainloop()
    return

def run_(file, filePath):
    """ call neccesary functions 
    """
    #make sure there are no existing figures open
    plt.close('all')

    # set location of flight data as surrent working directory
    os.chdir(filePath)
    preprocessDataResample(file, flightData, spatialResolution, lowcut, highcut, order)
    
    
        
    if siftThruHodo:
       manualTKGUI()
       
    if analyze:
        hodo_list= doAnalysis(microHodoDir)
        
        if showVisualizations:
            macroHodo()
            uvVisualize()
            plotBulkMicros(hodo_list, file)
        return
    return
     
#Calls run_ method  
run_(fileToBeInspected, flightData) 
    
#last line