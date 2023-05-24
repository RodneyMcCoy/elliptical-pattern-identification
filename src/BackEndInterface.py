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
from pathlib import Path
import main


# %% Back End Interface

def OpenBackendProcess(*args):
    """ The Main Secondary Process Controller. """
    frontend, currently_processing, files = args
    number = 0
    for file in files:
        if not currently_processing.is_set():
            break
        
        # Check If Processed Data For This File Can Be Found
        skip = False
        for data in os.listdir(main.DataOutputPath):
            # print(str(main.DataOutputPath / Path(data)), file + "_output")
            if str(main.DataOutputPath / Path(data)) == file + "_output":
                skip = True
                break
        if skip:
            print(file, " was already processed")
            continue
        
        # Continue To Process This File
        text = str(number) + " Files Processed: Now Processing "
        abs_path,file_name = os.path.split(file)
        text += file_name
        frontend.send(text)
        ProcessSingleFile(file)
        number += 1
        
    if number == 0:
        print("File Processing Was Started, But There Was No File Which Could Be Processed.")
        print("Either No Files Were Inputted, Or Every File That Was Found Already Had Outputted Data")

    currently_processing.clear()
    
    print("Processing Has Finished")
    return



def ProcessSingleFile(file):
    """ Here Is The Interface Of The Frontend To The Backend. 
    File Is An Absolute Filepath To The File We Want To Process. ALl That Is 
    Done Here Is Applying The Backend To That File And Saving The Results To A 
    Folder In DataOutputPath. """
    
    # XXX: Call The Backend On The File "file"
    backend.flightData, backend.fileToBeInspected = os.path.split(file)
    backend.main()

    return