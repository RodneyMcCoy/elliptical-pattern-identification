
# -*- coding: utf-8 -*-
"""
PDAP (Profile Data Analysis Program)

Author: Alex Chambers-Idaho

Created on Thursday July 16 13:30:48 2020
Last Edit: 10/9/2020 -  11:07 AM


PDAP DEVELOPER: -----------ALEX CHAMBERS - IDAHO
Program functions:
    -Rise Rate Analysis-----------------------------(Alex Chambers-Idaho)
    -Altitude Analysis------------------------------(Alex Chambers-Idaho)
    -RI-Planetary Boundary Layer--------------------(Hannah Woody & Keaton Blair-Montana)
    -VPT-Planetary Boundary Layer-------------------(Hannah Woody & Keaton Blair-Montana)
    -PT-Planetary Boundary Layer--------------------(Hannah Woody & Keaton Blair-Montana)
    -SH-Planetary Boundary Layer--------------------(Hannah Woody & Keaton Blair-Montana)
    -Radiosonde Geo-tracking------------------------(Lauren Perla & Alex Chambers- Idaho)
    -Lapse Rate Calculations (NEEDS REFINEMENT)-----(Main: Shirley Davidson-Montana & Leah Davidson-Idaho;AWContributors for Python Version: Alex Chambers & Malachi Rivkin-Idaho)
    -Tropopause Incorperation-----------------------(Alex Chambers-Idaho)
    -Consecutive Profile Running--------------------(Keaton Blair-Montana)
    -Save to a predetermined location---------------(Keaton Blair-Montana)
    -Numerical Analysis saved to text file----------(Alex Chambers-Idaho)
                                                     
Program functions developed by members from 
    -University of Idaho
    -University of Montana
"""

import numpy as np                                    # Numbers (like pi) and math
import matplotlib.pyplot as plt                       # Easy plotting
import pandas as pd                                   # Convenient data formatting, and who doesn't want pandas
from numpy.core.defchararray import lower             # For some reason I had to import this separately
import os                                             # File reading and input
from io import StringIO                               # Used to run strings through input/output functions
from scipy import interpolate,optimize                # Used for PBL calculations
                                       # Used for the Geotrack for plotting in Google -- IF USING ANACONDA MUST DOWNLOAD
import sys                                            # Used to control entire program (ie. stop run)
from matplotlib.offsetbox import AnchoredText         # Used to add text boxes to plots
import tkinter as tk                                  # Used to create Window Explorer 
from tkinter import filedialog,Tk                     # Used to create Window Explorer

"""
Operational Instructions:
    
    1) If running Anaconda/Spyder, download indicated modules (If not Anaconda/Spyder, may need to download more than indicated)
    2) Place Profile Data into a designated folder (can have one or multiple)
    3) Run Program
    4) Windows Explorer should open- select folder containing profile to analyze
    5) Follow Instructions in python console for save features
        -If you want to save outputs, type 'Y' for yes
        -If save path should be same as folder profile came from, type 'Y' for yes
        -If you wish to save to different folder, type 'N' for no, and select folder to save to
    6) Program will run over the number of profiles in the selected folder, and save accordingly
    
    **NOTE: all graphs will include profile name in title,
            and all saved graphs and documents will include profile name in filename for easy identification
"""



def LapseRate(pblHeightPT,pblHeightRI,pblHeightSH,pblHeightVPT,saveData):
    global heightsList
    if type(pblHeightVPT) == str:
        heightsList = [pblHeightPT, pblHeightRI, pblHeightSH]
    else:
        heightsList = [pblHeightPT, pblHeightRI, pblHeightSH, pblHeightVPT]
   
    meanHeights = np.mean(heightsList)
    sd = np.std(heightsList)
    heightsList = [x for x in heightsList if (x > meanHeights -  sd)]
    heightsList = [x for x in heightsList if (x < meanHeights + sd)]
    
    cutoff = int(np.mean(heightsList))
    
    lr_hi = np.where(hi<cutoff)
    cutoff_ind = lr_hi[0]
    cutoff_ind = cutoff_ind[-1]
    print(hi[cutoff_ind+1])
    
    slope,intercept=np.polyfit(hi[0:cutoff_ind+1], vt[0:cutoff_ind+1], 1)
    fit = (slope*hi[0:cutoff_ind+1])+intercept
    f = plt.figure('lapseRate: {}'.format(file))
    
    plt.plot(hi[0:cutoff_ind+1], fit)
    plt.plot(hi[0:cutoff_ind+1], vt[0:cutoff_ind+1])
    plt.xlabel('Altitude (m)')
    plt.ylabel('Temperature (K)')
    plt.annotate('Lapse Rate (K/Km) = {0:.4f}'.format(slope*1000), (0.09, 0.05), xycoords='axes fraction')
    #plt.title('Lapse Rate Determination \n %.20s' %(file),fontsize=12)
    if saveData:
       plt.savefig('%s/LapseRate_TempVAlt_%.20s.jpg' %(savePath,file))
    plt.show()

    return slope*1000

def pblRI(vpt, u, v, hi):
    # The RI method looks for a Richardson number between 0 and 0.25 above a height of 800 and below a height of 3000
    # The function will then return the highest height of a "significant RI"
    # NOTE: these boundaries will need to be changed if you are taking data from a night time flight,
    # where the PBL can be lower than 800m
    # or a flight near the equator where the PBL can exceed a height of 3000m

    # Support for this method and the equation used can be found at the following link:
    # https://resy5.iket.kit.edu/RODOS/Documents/Public/CD1/Wg2_CD1_General/WG2_RP97_19.pdf
    # https://watermark.silverchair.com/1520-0469(1979)036_0012_teorwa_2_0_co_2.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAsMwggK_BgkqhkiG9w0BBwagggKwMIICrAIBADCCAqUGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMtQBGOhb8zchwxViIAgEQgIICduC_f9w94ccDO9Nz1u73Ti7uOmXyjo_dLzL6LsXhu0-0uMAxTRsrPuPu_aCgyt4vyLVccC1OeRc9KR5npTEGstzVFFZs-vFNNs8Bl78f1K5jOhlAT9DYH3oSp3vdEM763kaZDV_1mc-8QzJORohbeGB1YOu4TbqYd70ZoJCS59yKO7emrSfcVVdQIWNOQ6PoT4ONeDowOCXCIgv4WBO-ul9fKAuA217EvXIh3-5o_SGj-SuMO30ktr8htOstvD_dC36eB3efxJ9l2MyDwvurUAO4CfJBgpaCKAg4af8LeljpmlXbFgkB7_jQyVXYvdfZNxvjAmp72Nbn6x_qjRc3TMhrhzw4R0ZtwjF9IjfDz-zolAwDPZ_PALKP-HE-M-Zi7q9hRd6XxDsjVOINTpZ07apgpT0ssX58uU3aPAiWDZnEInwz2-r_b_6KJHABRFWj4GYmW34v35nQz_xCo20S3MRQ-Lh7CiiwIAvkchNIfpScUI11Kz7Hd8gLsVqQ7r8fp4iWbgc4NEkS2gRkj8XEIqdvvFyCLLPo6bs_20iVtyEuGuwWQM3fYbpiS38iqth9LFcx7suDYUbMd1GbrYR3gdbvr9KKLohN6-rCJV-8rxIDOqraPxewIJyOckPHEaQ5Ek1Q1FEahweLE3HDgz93DnDQHoHYrmDU0gmsvDRqtxVRnqVf95d3V5DQNom8MFPEZiRdv7Vb8-2BQq_GMYEXZrv0FeKVr40HLSRy5Kc4qXZBR97XjN04AEyJ-umhyrb5DuzQdksk2T5WTXIIlx3DmZXLYY5Ond0cXDhOjGh7A6sPiJ2jVPTEzwSdwXUtpnMdxdpsFX6GhQ

    g = 9.81  # m/s/s
    ri = (((vpt - vpt[0]) * hi * g) / (vpt[0] * (u ** 2 + v ** 2)))
   
    index = [ 0 <= a <= 0.25 and 800 <= b <= 3000 for a, b in zip(ri, hi)]     #Check for positive RI <= 0.25 and height between 800 and 3000 meters

   
    plt.plot(ri, hi)        #Plots the Ri number against height above ground in meters
    plt.xlim(-1, 1)
    plt.ylim(500, 2000)
    plt.plot([0.25] * 2, plt.ylim())    #Sets a vertical line at the critical Ri number of 0.25
    plt.xlabel("Ri Number")
    plt.ylabel("Height above ground in meters")
    plt.title('RI PLD Determination \n %.20s' %(file),fontsize=12)
    if saveData:
       plt.savefig('%s/RI_PLB_%.20s.jpg' %(savePath,file))
    plt.show()
 
    
    if np.sum(index) > 0:    #Return results if any found
        return np.max( hi[index] )

    # Otherwise, interpolate to find height
    index = [800 <= n <= 4000 for n in hi]   #Trim to interested range
    hi = hi[index]
    ri = ri[index]

     
    return np.interp(0.25, ri, hi)   #Interpolate, returning either 800 or 3000 if RI doesn't cross 0.25



def pblPT(hi, pot):
    global xxx
    # The potential temperature method looks at a vertical gradient of potential temperature, the maximum of this gradient
    # ie: Where the change of potential temperature is greatest
    # However, due to the nature of potential temperature changes you will need to interpret this data from the graph
    # produced by this function. In this case you are looking for a maxium on the POSITIVE side of the X=axis around the
    # height that makes sense for your predicted PBL height
    # NOTE: arrays at the top of the function start at index 10 to avoid noise from the lower indexes.

    # Support for this method can be found at the following link:
    # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009JD013680

   
    topH = 3000 # High and low height limits for the PBL
    lowH = 800

    # Trim potential temperature and height to within specified heights
    height3k = [i for i in hi if lowH <= i <= topH]
    pt = [p for p, h in zip(pot, hi) if lowH <= h <= topH]

    dp = np.gradient(pt, height3k)                                        #Creates a gradient of potential temperature and height
    plt.plot(dp, height3k)                                                #Creates plot to determine the PBL Height
    plt.ylim(800, 2000)                                                   #Change if believed PBL may be higher than 2000, or lower than 800
    plt.xlabel("Gradient of PT")
    plt.ylabel("Height above ground in meters")
    plt.title('Potential Temp PBL Determination \n %.20s' %(file),fontsize=12)
    if saveData:
       plt.savefig('%s/PT_PBL_%.20s.jpg' %(savePath,file))
    plt.show()
    
    arr = np.array(height3k)[dp == np.max(dp)]
   
    return  int("".join(map(str,arr)))   #Return height of maximum gradient


def pblSH(hi, rvv):
    # The specific humidity method looks at a vertical gradient of specific humidity, the minimum of this gradient
    # ie: where the change in gradient is the steepest in negative direction
    # However, due to the nature of specific humidity changes you will need to interpret this data from the graph
    # produced by this function. In this case you are looking for a maxium on the NEGATIVE side of the X=axis around the
    # height that makes sense for your predicted PBL height
    # NOTE: arrays at the top of the function start at index 10 to avoid noise from the lower indexes.

    # Support for this method can be found at the following link:
    # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009JD013680

    q = rvv / (1 + rvv)  # equation for specific humidity

    # High and low height limits for the PBL
    topH = 3000
    lowH = 800

    # Trim potential temperature and height to within specified heights
    height3k = [i for i in hi if lowH <= i <= topH]
    q = [q for q, h in zip(q, hi) if lowH <= h <= topH]

    dp = np.gradient(q, height3k)                                          #Creates a gradient of potential temperature and height
    plt.plot(dp, height3k)                                                 #Creates plot to determine the PBL Height
    plt.ylim(800, 2000)                                                    #Change if believed  PBL may be higher than 2000 or lower than 800
    plt.xlabel("Gradient of Specific Humidity")
    plt.ylabel("Height above ground in meters")
    plt.title('Specific Humidity PBL Determination \n %.20s' %(file),fontsize=12)
    np.array(height3k)[dp == np.max(dp)]
    if saveData:
       plt.savefig('%s/SH_PBL_%.20s.jpg' %(savePath,file))
    plt.show()
   
    arr = np.array(height3k)[dp == np.max(dp)]
    
    return int("".join(map(str,arr)))  # Return height at maximum gradient



def pblVPT(pot, rvv, vpt, hi):
    # The Virtual Potential Temperature (VPT) method looks for the height at which VPT is equal to the VPT at surface level
    # NOTE: The VPT may equal VPT[0] in several places, so the function is coded to return the highest height where
    # these are equal

    # Supoort for this method can be found at the following link:
    # https://www.mdpi.com/2073-4433/6/9/1346/pdf
    global groot, grootVal

    vert_ln = [vpt[1]]*2                                                       #Defines the starting value of VPT as reference
    
    negXlim = vert_ln[1]-15                                                    #Creates variable for -15 than vertical line VPT 
    posXlim = vert_ln[1]+15                                                    #Creates variable for +15 than vertical line VPT  
    
    vptCutOff = vpt[1] 
    g = interpolate.UnivariateSpline(hi, vpt - vptCutOff,s=0)                  #Smoothing function
    plt.plot(vpt, hi,color='red')                                              #Plots VPT values against height
    plt.plot(g(hi)+vptCutOff, hi) 
    plt.plot(vert_ln, plt.ylim())                                              #Plots a vertical line at the X-value of VPT[1]
    axes = plt.gca() 
    axes.set_xlim([negXlim,posXlim])                                           #Centers X-bounds about Vertical line, easy to view
    axes.set_ylim([0,3000])                                                    #Average range to see values
    plt.xlabel("VPT")
    plt.ylabel("Height above ground [m]")
    plt.title('VPT PBL Determination \n %.20s' %(file),fontsize=12)
    if saveData:
       plt.savefig('%s/VPT_PBL_%.20s.jpg' %(savePath,file))
    plt.show()
    print ('Vertical Line for VPT:')
    print(vert_ln)                                                             #Feeds back the vertical line created by VPT[1] 
    
   
    rootdos = []
#The following section finds locations where VPT crosses the value of VPT[1]
    if len(g.roots()) == 0: 
        rootdos = "Error in Calculating VPT Method"                                          
        return rootdos
    else: groot = pd.DataFrame(g.roots())
    groot.columns=['Roots']
    if len(groot) >=1:
        grootVal = groot['Roots'].iloc[-1]
        grootVal = round(grootVal,3)
        grootVal = int(grootVal)
        if grootVal < 100:
             rootdos = "VPT Method Inconclusive-No Value Output in "
             return rootdos
        if grootVal <500 and grootVal >=100:
            rootdos = ("VPT Method Inconclusive-Estimated Value: %.5s" %(grootVal))
            return rootdos
        if grootVal >=500 and grootVal <=3000:
            rootdos = ("%.5s" %(grootVal))
            rootdos = int(rootdos)
            return rootdos
        if grootVal > 3000:
            rootdos = ("VPT Method Inconclusive-Estimated Value Beyond Nominal Range: %.5s" %(grootVal))
            return rootdos

    return rootdos           #Returns the highest ALT where VPT = VPT[1]






def layerStability(hi, pot):
# This function looks through potential temperature data to determine layer stability into 3 catergories
# NOTE: It is best to choose the higest PBL calculation unless the methods produce PBL Heights more than 300m
# apart. Also, when a stable boundary layer is detected, reject a PBL that is above 2000m, as these are often 
# night-time layers and a PBL near 2000m does not make sense
    ds = 1
    try:
        diff = [pot[i] for i in range(len(pot)) if hi[i] >= 150]
        diff = diff[0]-pot[0]
    except:
        return "Unable to detect layer stability, possibly due to corrupt data"

    if diff < -ds:
        return "Detected convective boundary layer"
    elif diff > ds:
        return "Detected stable boundary layer"
    else:
        return "Detected neutral residual layer"
    

def getUserInputFile(prompt):
    print(prompt)
    main = Tk()                                   
    userInput = filedialog.askdirectory()     #Creates directory for user to choose (location of profile data)
    main.destroy()
    if userInput == "":                       #If user cancels or does not select, exit the program
        sys.exit()
    return userInput                          #Return the file directory user chose


def getUserInputTF(prompt):
    print(prompt+" (Y/N)")                   #Prompts user for a Yes or No 
    userInput = ""
    while not userInput:
        userInput = input()
        if lower(userInput) != "y" and lower(userInput) != "n":
            userInput = "Please enter a 'Y' or 'N'"
    if lower(userInput) == "y":
        return True
    else:
        return 


dataSource = getUserInputFile("Select path to data input directory: ")   #File directory location
saveData = getUserInputTF("Do you want to save the output data?")        #Save location
if saveData:
    SavePrompt = getUserInputTF("Save to same directory?")                  #If save selected...
    if SavePrompt:                                                          #If yes, save to folder you pulled from?
        savePath = dataSource
    elif saveData:
        savePath = getUserInputFile("Enter path to data output directory:") #If no, ask for folder location to save
else:
    savePath = "NA"                                                        #If not save, make save path empty
    

# For debugging, print results
print("Running with the following parameters:")
print("Path to input data: /"+dataSource+"/")
print("Save data: "+str(saveData))

########## FILE RETRIEVAL SECTION ##########

# Need to find all txt files in dataSource directory and iterate over them

for file in os.listdir(dataSource):
    if file.endswith(".txt"):

        #Used to fix a file reading error
        contents = ""
        #Check to see if this is a GRAWMET profile
        isProfile = False
        f = open(os.path.join(dataSource, file), 'r')
        print("\nOpening file "+file+":")
        for line in f:
            if line.rstrip() == "Profile Data:":
                isProfile = True
                contents = f.read()
                print("File contains GRAWMET profile data")
                break
        f.close()
        if not isProfile:
            print("File "+file+" is either not a GRAWMET profile, or is corrupted.")

        if isProfile:  # Read in the data and perform analysis

            # Fix a format that causes a table reading error
            contents = contents.replace("Virt. Temp", "Virt.Temp")
            contents = contents.split("\n")
            contents.pop(1)  # Remove units from temp file
            index = -1
            for i in range(0, len(contents)):  # Find beginning of footer
                if contents[i].strip() == "Tropopauses:":
                    index = i
            for item in contents:              #Find Tropopause string in footer
                if '1. Tropopause:' in item:
                    trop = (item.strip())
            if index >= 0:  # Remove footer, if found
                contents = contents[:index]
            contents = "\n".join(contents)  # Reassemble string
            del index

            # Read in the data
            print("Constructing a data frame")
            RawData = pd.read_csv(StringIO(contents), delim_whitespace=True,na_values=['-'])
            del contents

            # Find the end of usable data
            badRows = []
            for row in range(RawData.shape[0]):
                if not str(RawData['Rs'].loc[row]).replace('.', '', 1).isdigit():  # Check for nonnumeric or negative rise rate
                    badRows.append(row)
                elif row > 0 and np.diff(RawData['Alt'])[row-1] <= 2:
                    badRows.append(row)
                else:
                    for col in range(RawData.shape[1]):
                        if RawData.iloc[row, col] == 999999.0:  # This value appears a lot and is obviously wrong
                            badRows.append(row)
                            break
            if len(badRows) > 0:
                print("Dropping "+str(len(badRows))+" rows containing unusable data")
            data = RawData.drop(RawData.index[badRows])         #Create dataframe of cleaned data
         
            
            ########## PERFORMING ANALYSIS ##########
            
            
            print("Starting Data Analysis")
            
            
############# The following are all the variables and equations needed for predicting PBL Height###########
            
            hi = data['Alt'] - data['Alt'][0]  # height above ground in meters
            epsilon = 0.622  # epsilon, unitless constant
            dwPt = data['Dewp.']
            
            #vapor pressure
            e = np.exp(1.8096+(17.269425*dwPt)/(237.3+dwPt))

            # water vapor mixing ratio
            rvv = (epsilon * e) / (data['P'] - e)  # unitless

            # potential temperature
            pot = (1000.0 ** 0.286) * (data['T'] + 273.15) / (data['P'] ** 0.286)  # kelvin

             
            virtcon = 0.61 # unitlessG:\Solar Eclipse  Project\PBL\2020
               
            # virtual potential temperature    
            vpt = pot * (1 + (virtcon * rvv))  # kelvin

            # absolute virtual temperature
            vt = (data['T'] + 273.15) * ((1 + (rvv / epsilon)) / (1 + rvv))  # kelvin

            # u and v (east & north?) components of wind speed
            u = -data['Ws'] * np.sin(data['Wd'] * np.pi / 180)
            v = -data['Ws'] * np.cos(data['Wd'] * np.pi / 180)
            

            
################# The following are variables adn equations needed for Lapse Rate Calculations#######
            dataLP = data[['Time', 'Virt.Temp', 'Alt']]
            dataLP['Altn'] = dataLP['Alt']/1000
            num_Data = dataLP.to_numpy()                #Create numpy array from df for use in LP Calculations
           
            
            ## CALL THE FUNCTIONS YOU WANT TO USE
            #Only functions that are called in this section will display in the output data
            
            
            pblHeightRI  = pblRI(vpt, u, v, hi)
            pblHeightVPT = pblVPT(pot, rvv, vpt, hi)
            pblHeightPT  = pblPT(hi, pot)
            pblHeightSH  = pblSH(hi, rvv)
            #Lapse_Rate   = LapseRate(pblHeightPT,pblHeightRI,pblHeightSH,pblHeightVPT,saveData)
            

   
            

        
 

print("\nAnalyzed all .txt files in folder /"+dataSource+"/") 


