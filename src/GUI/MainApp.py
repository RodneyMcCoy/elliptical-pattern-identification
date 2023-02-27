"""
Rodney McCoy
rbmj2001@outlook.com
November 2022
"""

# Contains the class which controls the Tkinter frontend

import sys
import os
import tkinter as tk
from pathlib import Path
import tkinter.filedialog

import FrameClasses as Frame






# Main Tkinter Application. Controls the tkinter gui along with communications between each frame of the GUI
class MainApp:
    
    def __init__(self, master):
        # INITIALIZE TKINTER GUI
        
        # Stores the inputted tkinter.Tk() window
        self.master = master
        
        # Assign title to window
        self.master.title("Ellipitical Pattern Identification")

        # Control intial size of window
        self.master.state('zoomed')
        self.master.geometry("800x500")

        # Set window icon
        # name = "logo.ico"
        # for root, dirs, files in os.walk(path):
        #     if name in files:
        #         return os.path.join(root, name)
        path = Path(os.getcwd()).parent.parent.absolute() / "res" / "logo.ico"
        
        try:
            self.master.iconbitmap(path)
        except tk.TclError:
            pass

        # Configure rows and cols
        self.master.rowconfigure(0, minsize=800, weight=1)
        self.master.columnconfigure(1, minsize=800, weight=1)
        
        # Clean exit when window is closed
        self.master.protocol("WM_DELETE_WINDOW", self.close)




        # INTIALIZE INTERNAL FILE STORAGE DATA STRUCTURE

        # Data Structure to Store Files
        self.file_container = []
        
        # Denotes which file is selected and should be displayed
        self.current_file_index = 0
        
        self.currently_processing = False
                
                
      
        
        # INITALIZE ALL FRAMES FROM THEIR CLASSES IN "FrameClasses.py"
        
        self.progress_window = Frame.ProgressWindow(self)
        self.search_window = Frame.SearchWindow(self)
        self.main_window = Frame.MainWindow(self)
        self.sidebar = Frame.Sidebar(self)
        self.file_window = Frame.FileWindow(self)

        # Ensure the starting window is the main one        
        self.switch_to_Main_Window()
        return

    

    # Purpose: Clean Exit of TKINTER GUI
    def close(self):
        # BUG weird errors thrown when code ran
        self.master.destroy()
        sys.exit()
        return



    # Purpose: Button command for adding a file to file_container
    def addFile(self):
        # Directory the file dialog window starts from
        initialdir = Path(os.getcwd()).parent.parent.absolute()

        # Tkinter command to get file as input from user
        filename = tk.filedialog.askopenfilename(
            initialdir = initialdir, title = "Select a File", filetypes = 
            (("Text files", "*.txt*"), ("all files", "*.*")))
         
        # If this is the first file inputted, activate buttons which need files
        if self.file_container == []:
            self.main_window.process_button.configure(state = tk.NORMAL)
            self.sidebar.buttons["next"].configure(state = tk.NORMAL)
            self.sidebar.buttons["previous"].configure(state = tk.NORMAL)
            self.sidebar.buttons["select"].configure(state = tk.NORMAL)


        # Add the inputted file to "file_container" if its not already in it
        add_file = True
        for file in self.file_container:
            if file.name == filename:
                add_file = False
        if add_file:
            self.file_container.append(filename)

        # Update "main" frame with info about new files
        self.main_window.file_label.configure(text="".join(
            [" " + f + " " for f in self.file_container])) 
        return
    


    # Purpose: Button command for adding a folder of files to file_container
    def addFolder(self):
        # Directory the file dialog window starts from
        initialdir = Path(os.getcwd()).parent.parent.absolute()
        
        # Tkinter command to get folder as input from user
        foldername = tk.filedialog.askdirectory(
            initialdir = initialdir, title = "Select a Folder")
         
        # If this is the first file inputted, activate buttons which need files
        if self.file_container == []:
            self.main_window.process_button.configure(state = tk.NORMAL)
            self.sidebar.buttons["next"].configure(state = tk.NORMAL)
            self.sidebar.buttons["previous"].configure(state = tk.NORMAL)
            self.sidebar.buttons["select"].configure(state = tk.NORMAL)
        
        # Add the inputted files to "file_container" if they are not already in it
        for filename in os.listdir(foldername):
            if os.path.splitext(filename)[-1].lower() == ".txt":
                add_file = True
                for file in self.file_container:
                    if file == filename:
                        add_file = False
                if add_file:
                    self.file_container.append(filename)
            
        # Update "main" frame with info about new files
        self.main_window.file_label.configure(text="".join(
            [" " + f + " " for f in self.file_container])) 
        return



    # Purpose: Behavior of GUI for when we switch to the "progress window"
    def switch_to_Progress_Window(self):
        self.progress_window.load_this_frame()
        self.sidebar.state(False)
        return



    # Purpose: Behavior of GUI for when we switch to the "main window"
    def switch_to_Main_Window(self):
        self.main_window.load_this_frame()
        return
        
        
        
    # Purpose: Behavior of GUI for when we switch to the "search window"
    def switch_to_Search_Window(self):
        self.search_window.load_this_frame()
        return
        
        
        
    # Purpose: Behavior of GUI for when we switch to the "file window"
    def switch_to_File_Window(self, index):
        self.file_window.load_this_frame(index % len(self.file_container))
        return
        
        
    
    # Purpose: Behavior of GUI for when next file button is pressed
    def next_file(self):
        self.current_file_index = (self.current_file_index + 1) % len(self.file_container)
        self.switch_to_File_Window(self.current_file_index)
        return
        
        
        
    # Purpose: Behavior of GUI for when previous file button is pressed
    def previous_file(self):
        self.current_file_index = (self.current_file_index - 1) % len(self.file_container)
        self.switch_to_File_Window(self.current_file_index)
        return