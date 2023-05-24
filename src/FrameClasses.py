"""
Rodney McCoy
rbmj2001@outlook.com
December 2022

This script contains various classes inheriting from a Tkinter frame.
Each controls behavior for a specific window in the front end.
For example, Sidebar controls the buttons on the left column, and ProgressWindow
is activated when files are currently being processed.

These classes communicate with the MainApp class.
"""

# Standard Python Libraries
import os

# Graphics and File Processing
import tkinter as tk
import tkinter.ttk as ttk
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Files In Code Base
import MainApp
import main


# %% Sidebar Frame

class Sidebar(tk.Frame):
    """ Controls Behavior Of Widgets On The Sidebar / Left Hand Column Of The 
    GUI. It Mainly Contains Buttons, And So It Communicates Alot With The 
    MainApp And Other Frame Classes. As Opposed To The Other FrameClasses, This
    One Is Always Activated. """
    
    def __init__(self, app : MainApp):
        # Save Inputed MainApp class. Build Sidebar Frame.
        self.app_reference = app
        self.master = app.master
        self.frame = tk.Frame(self.master, relief=tk.RAISED, bd=2)

        # Create Button Widgets For This Frame.
        self.buttons = {}
        
        self.buttons["close"] = ttk.Button(self.frame, text="Quit", 
            command = self.app_reference.close)
        self.buttons["main"] = ttk.Button(self.frame, text="Main", 
            command = self.app_reference.switch_to_Main_Window)
        self.buttons["select"] = ttk.Button(self.frame, text="Select A File", state = tk.DISABLED,
            command = self.app_reference.switch_to_Search_Window)
        self.buttons["previous"] = ttk.Button(self.frame, text="Go to Previous File", state = tk.DISABLED,
            command = self.app_reference.previous_file)     
        self.buttons["next"] = ttk.Button(self.frame, text="Go to Next File", state = tk.DISABLED,
            command = self.app_reference.next_file)
        
        # Place Button Widgets Onto The Frame.
        for index, key in enumerate(self.buttons):
            self.buttons[key].grid(row=index, column=0, sticky="ew", padx=5, pady=8)

        # Place Sidebar Frame Onto The Master Window
        self.frame.grid(row=0, column=0, sticky="ns")
        return
    
    
    
    def state(self, activate : bool):
        """ Turns On or Off all of the Sidebar Buttons """
        if activate:
            for x in self.buttons:
                self.buttons[x].configure(state = tk.NORMAL)
        else:
            for x in self.buttons:
                self.buttons[x].configure(state = tk.DISABLED)
        return
    
        

# %% Main Window Frame

class MainWindow(tk.Frame):
    """ This Class And Frame Controls The Right Side Of GUI On Startup And When
    Main Button Is Pushed. It Includes Behavior For File Input And Initializing
    The File Processing. """
    
    def __init__(self, app : MainApp):
        # Save Inputed MainApp class. Build Main Window Frame.
        self.app_reference = app
        self.master = app.master
        self.frame = tk.Frame(self.master, relief=tk.RAISED, bd=0)

        # Create Widgets for this Frame.
        self.file_button = ttk.Button(self.frame, text="Open a File",
            command = self.app_reference.addFile)
        self.folder_button = ttk.Button(self.frame, text="Open all Files in a Folder",
            command = self.app_reference.addFolder)
        self.process_button = ttk.Button(self.frame, text="Process Files", state = tk.DISABLED,
            command = self.app_reference.switch_to_Progress_Window)
        self.file_label = ttk.Label(self.frame, text="Selected files will be placed here")

        
        # Place Widgets Onto This Frame.
        self.file_button.pack()
        self.folder_button.pack()
        self.process_button.pack()
        self.file_label.pack()

        # Place Main Window Frame Onto Master Window.
        self.frame.grid(row=0, column=1, sticky="nsew")
        return
        
        

    def load_this_frame(self):
        """ Called To Activate The Main Window On Startup Or Button Push. """
        self.frame.tkraise()
        return



# %% Progress Window Frame

# Class for initialzing and controlling some behavior of the Progress Window Frame
class ProgressWindow(tk.Frame):
    """ This Class and Frame Controls The Right Side Of GUI When Files Are 
    Being Processed. All Sidebar Buttons Will Deactivate And When Pressing
    Single Button, It Will Stop The File Processing. """
    
    def __init__(self, app):
        # Save Inputed MainApp class. Build Progress Window Frame.
        self.app_reference = app
        self.master = app.master
        self.frame = tk.Frame(self.master)
        
        # Create Widgets For This Frame.
        self.label = ttk.Label(self.frame, text=
            "Starting...")
        self.stop_button = ttk.Button(self.frame, text="Stop processing after this file", 
            command = lambda : self.app_reference.stop_processing())
        
        # Place Widgets Onto This Frame.
        self.label.pack()
        self.stop_button.pack()      
        
        # Place Progress Window Frame Onto Master Window.
        self.frame.grid(row=0, column=1, sticky="nsew")
        return
        
    def load_this_frame(self):
        """ Called To Activate The Progress Window On Button Push. """
        self.label["text"] = "Starting..."
        self.frame.tkraise()
        return

    def UpdateProcessing(self, *args):
        """ Used To Pass Information On Current Progress of File Processing. """
        self.label["text"] = "Processing File: " + str(args)



# %% Search Window Frame

# Class for initialzing and controlling some behavior of the Search Window Frame
class SearchWindow(tk.Frame):
    """ This Classs / Frame Controls The Right Side Of GUI When When Files Are 
    Being Searched. Contains Dropdown Which Shows All Files Currently Inputted. """
    
    def __init__(self, app):
        # Save Inputed MainApp Class. Build Progress Window Frame.
        self.app_reference = app
        self.master = app.master
        self.frame = tk.Frame(self.master)
        
        # Create Widgets For This Frame.
        self.options = []
        self.value = tk.StringVar()
        self.value.set( "Select From Files" )
        self.dropdown = ttk.Combobox(self.frame, value=self.options)
        self.dropdown.config(width = 100)
        self.submit_button = tk.Button(self.frame, text='Look At Data', command= self.submit_answer)

        # Place Widgets Onto This Frame.
        self.dropdown.pack()
        self.submit_button.pack()
        
        # Place The Frame Onto The Master Window.
        self.frame.grid(row=0, column=1, sticky="nsew")
        return
    
        

    def submit_answer(self):
        """ Controls The Behavior For When The Submit Button Is Pushed. """
        answer = self.dropdown.get()
        
        if answer == "Select From Files" :
            self.app_reference.switch_to_File_Window(0)
        else:
            for i, val in enumerate(self.app_reference.file_container):
                if val == answer:
                    break
            self.app_reference.switch_to_File_Window(i)
            self.app_reference.current_file_index = i
        return



    def load_this_frame(self):
        """ Called To Activate The Search Window On Button Push. """
        # Refresh The Search Dropdown
        self.options = []
        for file in self.app_reference.file_container:
            self.options.append(file) 
        self.dropdown.configure(value=self.options )

        # Move This Frame To The Front
        self.frame.tkraise()
        return



# %% File Window Frame

# Class for initialzing and controlling some behavior of the File Window Frame
class FileWindow(tk.Frame):
    """ This Class / Frame Controls Right Side Of GUI When Viewing The Relevant
    Information For A Testdata File. """

    def __init__(self, app):
        # Save Inputed MainApp Class. Build Progress Window Frame.
        self.app_reference = app
        self.master = app.master
        self.frame = tk.Frame(self.master)
        
        # Create A Label With Info For This Frame
        self.label = ttk.Label(self.frame, text="_")
        
        # Place The Widgets Onto This Frame.
        self.label.pack(side = tk.LEFT )

# %% 

        # XXX: Other Backend Dependent Widgets Initialized Here. 
        # So We Can Fine Tune The Information Rendered On The File Window
        self.files = []
        self.current_ellipse = 0
        self.image = ttk.Label(self.frame, text="Heights: ")
        self.image.pack(side = tk.TOP)
        
# %%

        # Place The Frame Onto The Master Window.
        self.frame.grid(row=0, column=1, sticky="nsew")
        return
        
    

    def load_this_frame(self, index):
        """ Called To Activate The File View Window On Button Push. """
        
        # Refresh The Top Right File Location Indicators.
        files = self.app_reference.file_container
        ind = []
        
        for i in range(-2, 2+1):
            if i != 0:
                ind.append((index + i) % len(files))
            elif i == 0:
                ind.append(index)
        string = ""
        
        for i in range(5):
            string += "(" + str(ind[i]) + ") "
            if i == 2:
                string += "CURRENT: "
            abs_path,file_name = os.path.split(files[ind[i]])
            string += file_name + "\n"
            
        # Jank Meant To Properly Extract File Path
        abs_path,file_name = os.path.split(files[index])
        self.path_to_data = main.DataOutputPath / Path(os.path.splitext(file_name)[-2]+"_output")
        
        # Move This Frame To The Front.
        self.frame.tkraise()

        # If No Output Data Found, Indicate This And Return        
        if not os.path.exists(self.path_to_data):
            self.label.configure(text=string) 
            self.print_file_data(False)
            return
        string += "Found Outputted Data"
        self.label.configure(text=string) 
        self.print_file_data(True)

    def print_file_data(self, data_found : bool):
        if data_found:
            # XXX: PLACE ALL RELEVANT FILE DATA ONTO WINDOW HERE.
            self.files = os.listdir(self.path_to_data)      
            for file in self.files:
                break
                # f = open(this_path / Path(file), "r")
            heights = open(self.path_to_data / Path("Heights"), "r")
            self.image["text"] = "Heights: " + heights.read()
            heights.close()
        else:
            self.image["text"] = ""


        return