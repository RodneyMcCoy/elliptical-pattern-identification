# -*- coding: utf-8 -*-

"""
Background removal process invlving 2D-FFT

Malachi Mooney-Rivkin
Last Edit: 6/3/2021
Idaho Space Grant Consortium
moon8435@vandals.uidaho.edu
"""

#dependencies
import os
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime
from datetime import timedelta

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
#from skimage.measure import EllipseModel

#interpolation
from scipy.interpolate import griddata

#fft
import scipy.fft 


###############################BEGINING OF USER INPUT##########################

#variables that are specific to analysis: These might be changed regularly depending on flight location, file format, etc.
r"""
flightData = r"C:\Users\M\OneDrive - University of Idaho\%SummerInternship2020\%%CHIILE_Analysis_Backups\ChilePythonEnvironment_01112021\ChileData_012721\Tolten_01282021"             #flight data directory
fileToBeInspected = 'T26_1630_12142020_MT2.txt'                                                 #specific flight profile to be searched through manually
microHodoDir = r"C:\Users\M\OneDrive - University of Idaho\workingChileDirectory\Tolten\T26_all"  
#microHodoDir = r"C:\Users\Malachi\OneDrive - University of Idaho\workingChileDirectory\Tolten\T28"              #location where selections from GUI ard. This is also the location where do analysis looks for micro hodos to analysis
waveParamDir = r"C:\Users\M\OneDrive - University of Idaho\workingChileDirectory"     #location where wave parameter files are to be saved
flightTimesDir = r"C:\Users\M\OneDrive - University of Idaho\%SummerInternship2020\hodographAnalysis\Tolten"
flightTimes = r"Tolten_FlightTimes.csv"
"""

flightData = r"C:\Users\Malachi\OneDrive - University of Idaho\%SummerInternship2020\%%CHIILE_Analysis_Backups\ChilePythonEnvironment_01112021\ChileData_012721\Tolten_01282021"             #flight data directory
fileToBeInspected = 'T26_1630_12142020_MT2.txt'                                                 #specific flight profile to be searched through manually
microHodoDir = r"C:\Users\Malachi\OneDrive - University of Idaho\workingChileDirectory\Tolten\T26_all"  
#microHodoDir = r"C:\Users\Malachi\OneDrive - University of Idaho\workingChileDirectory\Tolten\T28"              #location where selections from GUI ard. This is also the location where do analysis looks for micro hodos to analysis
waveParamDir = r"C:\Users\Malachi\OneDrive - University of Idaho\workingChileDirectory"     #location where wave parameter files are to be saved
flightTimesDir = r"C:\Users\Malachi\OneDrive - University of Idaho\%SummerInternship2020\hodographAnalysis\Tolten"
flightTimes = r"Tolten_FlightTimes.csv"


p_0 = 1000 * units.hPa
spatialResolution = 5
##################################END OF USER INPUT######################

def preprocessDataResample(file, path, spatialResolution, lambda1, lambda2, order):
    #delete gloabal data variable as soon as troubleshooting is complete
    """ prepare data for hodograph analysis. non numeric values & values > 999999 removed, brunt-viasala freq
        calculated, background wind removed

        Different background removal techniques used: rolling average, savitsky-golay filter, nth order polynomial fits
    """
 
    data = openFile(file, path)
    data = interpolateVertically(data)
    
    #change data container name, sounds silly but useful for troubleshooting data-cleaning bugs
    global df
    df = data
    #print(df)
    #make following vars availabale outside of function - convenient for time being, but consider changing in future
    """
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
    
    """
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
    
        
    return 
def openFile(file, path):
    """
    open profile, package necessary data in dataframe
    """
    #indicate which file is in progress
    #print("Analyzing: {}".format(file))
    
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
    return data

def polarPlots():
    
    return

def interpolateVertically(data):
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
    return data

def f(t,y):
    """
    invent function to test plotting with
    """
    f_t = 1   #hz
    f_y = 2 #hz
    f_noise_height = 20 #Hz
    f_noise_time = 10 #Hz
    signal = np.sin(f_t*2*np.pi*t) * np.sin(f_y*2*np.pi*y)
    noise = .1 * (np.sin(f_noise_time*2*np.pi*t) * np.sin(f_noise_height*2*np.pi*y))
    plt.contourf(signal, levels=100)
    return signal + noise

"""
def constructBackGroundFFT(directory):
    for file in os.listdir(directory):
        #print(file)
        a=1
        
    print("HERE")
    xx = np.linspace(0,10*np.pi)
    yy = np.linspace(0,5*np.pi)
    xv, yv = np.meshgrid(xx, yy)
    print("length meshgrid: ", len(xv), "Length y:", len(yv))
    global z
    z = f(xv,yv)
    print(z)
    plt.imshow(z, interpolation='nearest')
    
    import scipy as sp
    fft = sp.fft.fft2(z)
    global freqs
    ###
    FreqCompRows = np.fft.fftfreq(FFTData.shape[0],d=2) # from internet...
    FreqCompCols = np.fft.fftfreq(FFTData.shape[1],d=2)
    ###
    freqs = sp.fft.fftfreq(np.size(z))
    print("Freqs: ", freqs)
    print("FFT: ", fft)
    
    
    fig, ax = plt.subplots()
    ax.contourf(fft)
    #xy = np.column_stack([xv,yv,z])
    #ax.contourf(xv,yv,z)
    #xy_fft = np.fft.fft2(xy)
    
    return
"""
def constructContourPlot(directory, times, timesPath):
    
    global x,y,z, xi, yi, zi, points, knownPoints, knownValues, grid, Xi, Yi, xi, yi, points, epoch, z_fft, z_filtered
    
    #retrieve flight times/dates from file; combine date, time columns into datetime object
    schedule = pd.read_csv(os.path.join(timesPath, times), skiprows=[1], parse_dates=[[2,3]])
    print(schedule)\
    
    #plot time series to confirm functionality
    #F, A = plt.subplots()
    #A.plot(schedule["Date_Time"], schedule["Number"])
    #A.set_xlabel("Time [UTC]")
    #date_form = DateFormatter("%H:%M")
    #A.xaxis.set_major_formatter(date_form)
    
    global bulkData     # useful for troubleshooting
    bulkData = pd.DataFrame()
    for file in os.listdir(directory):
        print(file)
        #print(type(file ))
        
        #get starting time of flight
        num = file.split("_")[0]    #get flight initial and number from file name
        #num = num[1:]   #remove flight initial - this results in flight number 
        num = [x for x in num if x.isdigit()]
        num = int("".join(num))
        
        if num < 50: #temporarily use first 4 profiles
            
            #print(type(num))
            #print("num: ", num)
            startTime = schedule.loc[schedule["Number"] == num, 'Date_Time'].values[0]
            print("Start Time: ", startTime)
            print("Start Time Type: ", type(startTime))
            
            
            #assign proper timestamp to each data entry in file
            data = openFile(file, directory)
            data = interpolateVertically(data)
        
            
            data["Time"] = pd.to_timedelta(data["Time"], unit='seconds')    #convert seconds into timedelta type
            data["Time"] = data["Time"] + startTime
            #print(data)
            #add to time series of all profiles
            bulkData = bulkData.append(data, ignore_index=True) #ignore index necessary?
            global trouble     # useful for troubleshooting
            trouble = data
            #print("Time Type: ", type(data["Time"]))
            
            
    print("BULK DATAFRAME")
    print(bulkData)
    
    #sort time series chronologically
    bulkData.sort_values(by=['Time'], inplace=True)
    #bulkData['Time'] = pd.to_datetime(bulkData['Time'])
    print("Bulk Data - chronological")
    print(bulkData)
    print("bulktime type: ", type(bulkData['Time'].iloc[1]))
    print("\n")
    
    #convert datetime to seconds since start of first launch
    bulkData['Time'] = bulkData['Time'] - bulkData['Time'].min()
    bulkData['Time'] = bulkData['Time'].dt.total_seconds()
    print("new time data type:",type(bulkData['Time'][0]))
    
    #interpolate bulk data onto grid -----------------------------------------------------------------
    ngridy = 1000   #need to decide on grid spacing
    ngridt = 1000
    grid = bulkData[['Time', 'Alt', 'T']]
    grid = grid.dropna()
    x = grid['Time']
    y = grid['Alt']
    knownPoints = (x,y)
    knownValues = grid['T']
    print("Size of knownvals: ",np.size(knownValues))
    print("Size of x: ",np.size(x))
    print("Size of y: ",np.size(y))

    # Create grid values first.
    xi = np.linspace(min(bulkData['Time']), max(bulkData['Time']), ngridt)
    yi = np.linspace(min(y), max(y), ngridy)
    
    #cordinates of grid
    Xi, Yi = np.meshgrid(xi, yi)
    points = (Xi,Yi)
    
    #interpolate onto grid
    #this method is effective
    #zi = scipy.interpolate.griddata(knownPoints, knownValues, points, method='linear')
    zi = scipy.interpolate.griddata(knownPoints, knownValues, points, method='linear')#, fill_value='extrapolate')
    #print("Starting INTERPOLATION")
    #f = scipy.interpolate.interp2d(x, y, knownValues, bounds_error=False)
    #print("Still working on INTERPOLATION")
    #zi = f(xi,yi)
    #print("FINISHED INTERPOLATION")
    #print(zi)


    global notnanIndices, zi2, knownPoints2, knownValues2
    #notnanIndices = np.argwhere(~np.isnan(zi))
    #knownPoints2 = (pd.Series(notnanIndices[:,0]), pd.Series(notnanIndices[:,1]))
    #xIndices = knownPoints2[0]
    #yIndices = knownPoints2[1]

    #knownValues2  = zi[xIndices,yIndices]
    
    #zi2 = scipy.interpolate.griddata(knownPoints2, knownValues2, points, method='nearest')
    #indices of nans zi
    
    #replace nans with zeros
    #zi = np.nan_to_num(zi)
    
    #construct background
    #xy = np.column_stack([z, xv,yv])
    
    #apply 2d fft
    print("zi type: ", type(zi))
    z_fft = scipy.fft.fft2(zi)
    #z_shift = scipy.fft.fftshift(z_fft)
    print("fft signal type: ", type(z_fft))
    #shift fft
    #xy_fft = scipy.fft.fftshift(xy_fft)
    """
    #figure = plt.figure("Raw Signal")
    #plt.contourf(zi, levels=100)
    """
    #magnitude f 2d fft
    s_mag = np.abs(z_fft)
    
    timestep = (max(x)-min(x))/ngridt    #s/sample
    heightstep = (max(x)-min(x))/ngridy    #m/sample
    #calculate frequency compnents of each bin
    FreqCompTime = np.fft.fftfreq(zi.shape[0],d=timestep)
    FreqCompHeight = np.fft.fftfreq(zi.shape[1],d=heightstep)
    
    #block high freq in height
    #cutoff_height = 1/10    #wavelength [m] (1/f)
    cutoff_height = 35000
    #cutoff_time = 1/10  #wavelength [s] (1/f)
    cutoff_time = 60*60*24  #wavelength [s] (1/f)
    
    z_filtered = z_fft.copy()
    z_filtered.T[abs(FreqCompTime) >= 1/cutoff_time] = 0
    z_filtered.T
    z_filtered[abs(FreqCompHeight) >= 1/cutoff_height] = 0
    
    #invert fft
    z_filtered = scipy.fft.ifft2(z_filtered)
    
    #plotting
    #filtered 
    figure = plt.figure("Filtered Image")
    plt.contourf(Xi, Yi, z_filtered, levels=50)
    
    """
    fig, (ax0, ax1, ax2) = plt.subplots(1,3)
    ax0.contourf(xv,yv,z, levels=100)
    
    #magnitude
    ax1.plot(s_mag)
    ax1.set_title("magnitude")
    """
    #create figure for plotting interpolated data
    contour, ax1 = plt.subplots()
    ax1.contour(Xi, Yi, zi, linewidths=0.5, colors='k')
    cntr1 = ax1.contourf(xi, yi, zi, levels=100, cmap='rainbow')
    contour.colorbar(cntr1, ax=ax1)
    ax1.set_xlabel("Time (UTC) [ns - needs changed]")
    ax1.set_ylabel("Altitude (m)")
    #contour.colorbar(cntr1, ax=ax1)
    ax1.plot(x,y, 'ko', ms=.005)
    ax1.set_title("Contour Map Temp vs Alt vs Time")
    #ax1.set_title('grid and contour (%d points, %d grid points)' %(npts, ngridx * ngridy))
    plt.show()
    
    #create figure for background
    fig, ax = plt.subplots()
    cntr1 = ax.contourf(Xi, Yi, z_filtered, levels=100, cmap='rainbow')
    fig.colorbar(cntr1, ax=ax)
    
    #view convex hull
    """
    from scipy.spatial import ConvexHull, convex_hull_plot_2d
    points = points = np.column_stack((x,y))
    global hull
    hull = ConvexHull(points)
    
    #plt.plot(points[:,0], points[:,1], 'o')
    for simplex in hull.simplices:
        ax1.plot(points[simplex, 0], points[simplex, 1], 'k-')
        
    fig3, ax3 = plt.subplots()
    ax3.imshow(zi)
    fig3.show()
    ####
    
    
    
    fig, ax = plt.subplots()
    ax.tricontourf(bulkData['Time'],bulkData['Alt'],bulkData['T'])
    #ax.scatter(bulkData['Time'],bulkData['Alt'])
    ax.set_xlabel("Time [UTC]")
    date_form = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_form)
        
    
    """
    ###############################################################
    """
    #experiment to learn how to apply 2D FFT
    global xy_fft, FreqCompHeight, FreqCompTime, s_mag

    #invent some data

    xx = np.linspace(0,3, 100)
    yy = np.linspace(0,1.5, 100)
    #samplesx = xx.size[0]
    #samplesy = yy.size[1]
    xv, yv = np.meshgrid(xx, yy)
    print("length meshgrid: ", len(xv), "Length y:", len(yv))
    global z
    z = f(xv,yv)
    xy = np.column_stack([z, xv,yv])
    
    #apply 2d fft
    z_fft = scipy.fft.fft2(z)
    #z_shift = scipy.fft.fftshift(z_fft)
    print("fft signal type: ", type(z_fft))
    #shift fft
    #xy_fft = scipy.fft.fftshift(xy_fft)
    figure = plt.figure("Raw Signal")
    plt.contourf(z, levels=100)
    
    #magnitude f 2d fft
    s_mag = np.abs(z_fft)
    
    timestep = 3/100    #s/sample
    heightstep = 1.5/100    #m/sample
    #calculate frequency compnents of each bin
    FreqCompTime = np.fft.fftfreq(z.shape[0],d=timestep)
    FreqCompHeight = np.fft.fftfreq(z.shape[1],d=heightstep)
    
    #block high freq in height
    #cutoff_height = 1/10    #wavelength [m] (1/f)
    cutoff_height = 1/20
    #cutoff_time = 1/10  #wavelength [s] (1/f)
    cutoff_time = 1/20  #wavelength [s] (1/f)
    
    z_filtered = z_fft.copy()
    z_filtered.T[abs(FreqCompTime) >= 1/cutoff_time] = 0
    z_filtered.T
    z_filtered[abs(FreqCompHeight) >= 1/cutoff_height] = 0
    
    #invert fft
    z_filtered = scipy.fft.ifft2(z_filtered)
    
    #plotting
    #filtered 
    figure = plt.figure("Filtered Image")
    plt.contourf(z_filtered, levels=100)
    #original data
    #figure = plt.figure("Raw Image")
    #plt.contourf(z, levels=100)
    fig, (ax0, ax1, ax2) = plt.subplots(1,3)
    ax0.contourf(xv,yv,z, levels=100)
    
    #magnitude
    ax1.plot(s_mag)
    ax1.set_title("magnitude")
    
    #filtered height
    ax2.plot(z_filtered)
    #fft (time)
    #ax2.plot(s_mag.T)
    #shifted data
    #ax2.contourf(fft_shift)
    
    #frequency content
    #plt.figure("FFT")
    #plt.plot(xy_fft)
      
    #end experiment
    """
    ###########################################################
    
    return
#Run data to construct background
constructContourPlot(flightData, flightTimes, flightTimesDir)


#preprocessDataResample(fileToBeInspected, flightData, 5, 1, 1, 3)

#constructBackGroundFFT(flightData)