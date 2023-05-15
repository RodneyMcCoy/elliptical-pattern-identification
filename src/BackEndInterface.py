"""
Rodney McCoy
rbmj2001@outlook.com
May 2023

This Script Controls Behavior For The Secondary Process. This Is Essential Because
If We Call The Backend From The Same Process As The GUI, It Freezes Until
The Backend Is Done, And my Multithreaded Attempt Ended Up Making The App
Unusable, Tkinter Fails With Multithreading. All Direct Interfacing With The 
Backend Occurs Here.
"""

# Standard Python Libraries
import os

# Files In Code Base
import duttaHodograph_ellipse_identification as backend
import main


# %% Back End Interface

def OpenFileProcessingThread(*args):
    """ The Main Secondary Process Controller. """
    frontend, files = args
    for file in files:
        # Check If Frontend Says Processing Should Stop
        if frontend.poll():
            if frontend.recv() == "STOP":
                break
        # Check If Processed Data For This File Can Be Found
        for data in os.listdir(main.DataOutputPath):
            if data == file + "_output":
                print(file, " was already processed")
                continue
        frontend.send("Now Processing " + file)
        ProcessSingleFile(file)
    frontend.send("STOP")
    frontend.close()
    return



def ProcessSingleFile(file):
    """ Here Is The Interface Of The Frontend To The Backend. 
    File Is An Absolute Filepath To The File We Want To Process. ALl That Is 
    Done Here Is Applying The Backend To That File And Saving The Results To A 
    Folder In DataOutputPath. """
    
# %%
    
    # XXX: This Setup Works But Is Kind Of Janky Since I Just Wanted To Hook Up The 
    # Front End And Back End With The Minimal Amount Of Changes.
    backend.flightData, backend.fileToBeInspected = os.path.split(file)
    backend.main()

# %%

    return