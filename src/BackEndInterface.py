"""
Rodney McCoy
rbmj2001@outlook.com
May 2023

This Script Controls Behavior For The Secondary Thread. This Is Essential Because
If We Call The Backend From The Same Thread As The GUI, It Freezes Until
The Backend Is Done. All Direct Interfacing With The Backend Occurs Here.
"""

# Standard Python Libraries
import time, os, sys

# Files In Code Base
import duttaHodograph_ellipse_identification as backend


# %% Back End Interface

def OpenFileProcessingThread(*args):
    """ The Main Secondary Thread Controller. """
    pipe_to_frontend, file_container = args
    for file in file_container:
        pipe = ""
        if pipe_to_frontend.poll():
            pipe = pipe_to_frontend.recv()
            if pipe == "STOP":
                break
        pipe_to_frontend.send("Now Processing: " + file)
        ProcessSingleFile(file)
    pipe_to_frontend.send("File Processing Has Stopped")
    pipe_to_frontend.close()
    return



def ProcessSingleFile(file):
    """ Here Is The Interface Of The Frontend To The Backend. 
    File Is An Absolute Filepath To The File We Want To Process. ALl That Is 
    Done Here Is Applying The Backend To That File And Saving The Results To A 
    Folder In DataOutputPath. """
    
    # XXX: This Setup Works But Is Kind Of Janky Since I Just Wanted To Hook Up The 
    # Front End And Back End With The Minimal Amount Of Changes.
    backend.flightData, backend.fileToBeInspected = os.path.split(file)
    backend.main()
