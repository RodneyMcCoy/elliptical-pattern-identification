"""
Rodney McCoy
rbmj2001@outlook.com
November 2022
"""

import sys
import os
import tkinter as tk
import tkinter.ttk as ttk
from pathlib import Path

import FrameClasses as Frame
import FileData as File



class MainApp:
    
    def __init__(self, master):
        # Assign Inputed tk window to the class tk window
        self.master = master
        
        # assign title to the main window
        self.master.title("Ellipitical Pattern Identification")

        # control intial size of window
        self.master.state('zoomed')
        self.master.geometry("800x500")

        # set app icon
        path = Path(os.getcwd()).parent.parent.absolute() / "res" / "logo.ico"
        self.master.iconbitmap(path)

        # Data Structure to Store Files
        self.file_container = []
        
        # Denotes which file is selected and should be displayed
        self.current_file_index = 0
                
        # configure rows and cols
        self.master.rowconfigure(0, minsize=800, weight=1)
        self.master.columnconfigure(1, minsize=800, weight=1)
        
        
        self.master.protocol("WM_DELETE_WINDOW", self.close)
        
        
        # Initialize 4 different frames (sidebar, main, progress, file) using other classes
        self.progress_window = Frame.ProgressWindow(self)
        self.search_window = Frame.SearchWindow(self)
        self.main_window = Frame.MainWindow(self)
        self.sidebar = Frame.Sidebar(self)
        self.file_window = Frame.FileWindow(self)
        
        self.switch_to_Main_Window()
        

        
    

    # Purpose: Close window and python interpreter
    def close(self):
        # BUG weird errors thrown when code ran
        self.master.destroy()
        sys.exit()



    # Purpose: Add a single file to "files" list
    def addFile(self):
        initialdir = Path(os.getcwd()).parent.parent.absolute()
        # Get file from user
        filename = tk.filedialog.askopenfilename(
            initialdir = initialdir, title = "Select a File", filetypes = 
            (("Text files", "*.txt*"), ("all files", "*.*")))
         
        if self.file_container == []:
            self.sidebar.activate_buttons()
            self.main_window.activate_buttons()

        
        # Add File to "files" list
        add_file = True
        for file in self.file_container:
            if file.name == filename:
                add_file = False
        if add_file:
            self.file_container.append(File.FileData(name=filename))


        # Update GUI with new files
        self.main_window.file_label.configure(text="".join(
            [" " + f.get_label() + " " for f in self.file_container])) 



    # Purpose: Add all files in folder to "files" list
    def addFolder(self):
        initialdir = Path(os.getcwd()).parent.parent.absolute()
        
        # Get file from user
        foldername = tk.filedialog.askdirectory(
            initialdir = initialdir, title = "Select a Folder")
         
        if self.file_container == []:
            self.sidebar.activate_buttons()
            self.main_window.activate_buttons()
        
        # Add File to "files" list
        for filename in os.listdir(foldername):
            if os.path.splitext(filename)[-1].lower() == ".txt":
                add_file = True
                for file in self.file_container:
                    if file.name == filename:
                        add_file = False
                if add_file:
                    self.file_container.append(File.FileData(name=filename))
            
        
        # Update GUI with new files
        self.main_window.file_label.configure(text="".join(
            [" " + f.get_label() + " " for f in self.file_container])) 





    def switch_to_Progress_Window(self):
        self.progress_window.load_this_frame()

    def switch_to_Main_Window(self):
        self.main_window.load_this_frame()
        
    def switch_to_Search_Window(self):
        self.search_window.load_this_frame()
        
    def switch_to_File_Window(self, index):
        self.file_window.load_this_frame(index % len(self.file_container))
        
        
        
    def next_file(self):
        self.current_file_index = (self.current_file_index + 1) % len(self.file_container)
        self.switch_to_File_Window(self.current_file_index)
        
    def previous_file(self):
        self.current_file_index = (self.current_file_index - 1) % len(self.file_container)
        self.switch_to_File_Window(self.current_file_index)