"""
Rodney McCoy
rbmj2001@outlook.com
December 2022

This script is the main controller for the front end.
"""

# Standard Python Libraries
import os

# Graphics and File Processing
import tkinter as tk
from pathlib import Path

# Files In Code Base
import MainApp as App



# %% Main Function

DataInputPath = Path(os.getcwd()).parent.absolute() / "test"
DataOutputPath = Path(os.getcwd()).parent.absolute() / "processed data"



def main(): 
    """ Starts the GUI. """
    
    # Build The App
    master = tk.Tk()
    App.MainApp(master)
    
    # Read In The JSON File And Its Parameters
    
    # Start the Tkinter Main Loop
    master.mainloop()



# Call Main
if __name__ == '__main__':
    main()