"""
Rodney McCoy
rbmj2001@outlook.com
May 2023

This Script Controls Behavior For The Secondary Thread. This Is Essential Because
If We Call The Backend From The Same Thread As The GUI, It Freezes Until
The Backend Is Done. All Direct Interfacing With The Backend Occurs Here.
"""

# Standard Python Libraries
import time

# Files In Code Base
# import duttaHodograph_ellipse_identification as backend


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



# THIS IS THE FUNCTION WHICH GIVEN A FILE PATH, PROCESSES THE FUNCTION VIA THE BACKEND ALGORITHMS
def ProcessSingleFile(file):
    """ Here Is The Interfacing With The Backend. File Is An Absolute Filepath
    To The File We Want To Processing. ALl That Is Done Here Is Applying The
    Backend To That File And Saving The Results To A Folder In DataOutputPath. """
    time.sleep(.25)
    # backend.run_(file)
    # print(file)