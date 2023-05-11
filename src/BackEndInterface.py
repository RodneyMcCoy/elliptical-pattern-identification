"""
Rodney McCoy
rbmj2001@outlook.com
May 2023

This Script Controls Behavior For The Secondary Thread. This Is Essential Because
If We Call The Backend From The Same Thread As The GUI, It Freezes Until
The Backend Is Done. All Direct Interfacing With The Backend Occurs Here.
"""

# Standard Python Libraries
import time, os

# Files In Code Base
import duttaHodograph_ellipse_identification as backend


# %% Back End Interface

def OpenFileProcessingThread(*args):
    """ The Main Secondary Thread Controller. """
    MainAppReference = args[0]
    ContinueProcessing = args[1]
    FileContainer = args[2]
    for file in FileContainer:
        if not ContinueProcessing.is_set():
            break
        MainAppReference.progress_window.UpdateProcessing(file)
        ProcessSingleFile(file)
    MainAppReference.progress_window.stop_processing()
    MainAppReference.main_window.file_label.configure(text="Processing Has Stopped.")



def ProcessSingleFile(file):
    """ Here Is The Interface Of The Frontend To The Backend. 
    File Is An Absolute Filepath To The File We Want To Process. ALl That Is 
    Done Here Is Applying The Backend To That File And Saving The Results To A 
    Folder In DataOutputPath. """
    
    # XXX: This Setup Works But Is Kind Of Janky Since I Just Wanted To Hook Up The 
    # Front End And Back End With The Minimal Amount Of Changes.
    backend.flightData, backend.fileToBeInspected = os.path.split(file)
    backend.main()
