"""
Rodney McCoy
rbmj2001@outlook.com
November 2022

This script contains the object MainApp. It is the controller for the front end
and interfaces with the various frame classes as well as the second thread which
processes files.
"""

# Standard Python Libraries
import sys, os
import multiprocessing

# Graphics and File Processing
import tkinter as tk
from pathlib import Path
import tkinter.filedialog

# Files In Code Base
import FrameClasses as Frame
import BackEndInterface
import main



# %% Main App Class

class MainApp:
    """ This Class Is The Main Controller For The Tkinter GUI. It Communicates
    To And From Each FrameClass. And It Creates And Communicates With The 
    Secondary Thread Which Actually Processes The Files. """
    
    def __init__(self, master):
        """ Initialize The Tkinter GUI. """
        
        # Basic Tkinter Setup.
        self.master = master
        self.master.title("Ellipitical Pattern Identification")
        self.master.state('zoomed')
        self.master.geometry("800x500")

        # Find Logo File Path. 
        # TODO: Very Low Priority. This Doesn't Work After PyInstaller Bundles
        # The Application Since It Changes The Internal Directory Structure, 
        # Causing A Terminating Error When It Cant Find The Logo, the Try Except
        # Statement Should Make It Ignore The Error.
        path = Path(os.getcwd()).parent.parent.absolute() / "res" / "logo.ico"
        try:
            self.master.iconbitmap(path)
        except tk.TclError:
            pass

        # Configure Rows and Cols.
        self.master.rowconfigure(0, minsize=800, weight=1)
        self.master.columnconfigure(1, minsize=800, weight=1)
        
        # Clean Exit When Window Is Closed. 
        # TODO: Medium Priority. When Attempting To Exit The Window,
        # Python Terminal May Not Exit Cleanly. This Is An Attempt To Fix It.
        # I Dont Think It Breaks Anything, But I Commonly Get Errors Pertaining
        # To WM_DELETE_WINDOW, So This Fix Probably Needs More Thought.
        self.master.protocol("WM_DELETE_WINDOW", self.close)

        # Initialize The Internal File Storage List, self.file_container.
        self.file_container = []
        # Denotes Which File Is Selected And Should Be Displayed For FileWindow.
        self.current_file_index = 0
                
        # Initialize Each Class / Frame In "FrameClasses.py".
        self.progress_window = Frame.ProgressWindow(self)
        self.search_window = Frame.SearchWindow(self)
        self.main_window = Frame.MainWindow(self)
        self.sidebar = Frame.Sidebar(self)
        self.file_window = Frame.FileWindow(self)
        
        # Initialize Multiprocessing Objects To Communicate With Backend
        self.backend, self.frontend = multiprocessing.Pipe(duplex = True)
        self.currently_processing = multiprocessing.Event()
        self.currently_processing.clear()
        
        # Ensure The Starting Class / Frame Is MainWindow.
        self.switch_to_Main_Window()
        
        # Add Files Contained In Default Folders
        self.input_files(os.listdir(main.DataInputPath), str(main.DataInputPath))

        # for output_file in os.listdir(main.DataOutputPath):

        if self.file_container != []:
            self.files_are_inputted()

        return

    

    def close(self):
        """ Controls When Tkiner WM_DELETE_WINDOW Is Raised. Meant To Cleanly
        Exit Python Interpreter. """
        if self.currently_processing.is_set():
            self.stop_processing(True)
        self.master.destroy()
        sys.exit()
        return



    def addFile(self):
        """ Controls Button Behavior To Input A File. """
        
        # Directory The File Dialog Window Starts From
        initialdir = main.DataInputPath

        # Tkinter Command To Get File As Input From User.
        filename = tk.filedialog.askopenfilename(
            initialdir = initialdir, title = "Select a File", filetypes = 
            (("Text files", "*.txt*"), ("all files", "*.*")))
        
        if len(filename) == 0:
            self.main_window.file_label.configure(text="No File Was Selected.")
            return

        
        # If This Is The First File Inputted, Activate Buttons Which Assume
        # Files Are Already Given.
        if self.file_container == []:
            self.files_are_inputted()

        self.input_files([filename])
        return
    


    def addFolder(self):
        """ Controls Button Behavior To Input A Folder Full Of Files. """

        # Directory The File Dialog Window Starts From
        initialdir = main.DataInputPath
        
        # Tkinter Command To Get Folder As Input From User.
        foldername = tk.filedialog.askdirectory(
            initialdir = initialdir, title = "Select a Folder")
        
        if len(foldername) == 0:
            self.main_window.file_label.configure(text="The Inputted Folder Was Empty Or No Folder Was Selected.")
            return
                
        # If This Is The First File Inputted, Activate Buttons Which Assume
        # Files Are Already Given.      
        if self.file_container == []:
            self.files_are_inputted()
        
        self.input_files(os.listdir(foldername), foldername)
        return



    def input_files(self, files, folder = None):
        count = 0
        FoundProperFile = False
        for filename in files:
            if folder is None:
                this_path = Path(filename)
            else:
                this_path = Path(folder) / Path (filename)
            # Add The Inputted Files To "file_container" If Its Not Already In It
            if os.path.splitext(filename)[-1].lower() == ".txt":
                FoundProperFile = True
                if not str(this_path) in self.file_container:
                    self.file_container.append(str(this_path))
                    count += 1
            
        if FoundProperFile == False:
            self.main_window.file_label.configure(text="No file of proper formatting (ending in .txt) was found.")
            return
        
        # Update MainWindow With Info About New Files.
        Text = str(count) + " new files were inputted.\n"
        # Text += "".join([" " + f + "\n " for f in files])
        self.main_window.file_label.configure(text=Text) 
        return
    
    

    def files_are_inputted(self):
        """ Allows GUI Behavior Which Assumes Files Are Given """
        self.main_window.process_button.configure(state = tk.NORMAL)
        self.sidebar.buttons["next"].configure(state = tk.NORMAL)
        self.sidebar.buttons["previous"].configure(state = tk.NORMAL)
        self.sidebar.buttons["select"].configure(state = tk.NORMAL)
        return
    
        
        
    def switch_to_Progress_Window(self):
        """ Main Controller Of Behavior For When Activating The ProgressWindow """
        # Load Progress Window.
        self.progress_window.load_this_frame()
        
        # Freeze Buttons In Sidebar Window.
        self.sidebar.state(False)
        
        # Create Secondary Process For Using Backend.
        self.currently_processing.set()
        self.process = multiprocessing.Process(target=BackEndInterface.OpenBackendProcess,
                                               args= (self.frontend, self.currently_processing, self.file_container))

        self.is_processing_done()
        self.process.start()
        return



    def is_processing_done(self):
        if not self.currently_processing.is_set():
            self.stop_processing()
            return
        if self.backend.poll():
            pipe = self.backend.recv()
            print(pipe)
            self.progress_window.label["text"] = pipe
        
        self.master.after(250, self.is_processing_done)
        return


        
    def stop_processing(self):
        """ For When Processing Needs To Stop. Deactivates / Deletes Secondary
        Process. """
        self.sidebar.state(True)
        self.switch_to_Main_Window()
        self.currently_processing.clear()
        self.main_window.file_label["text"] = "Processing Has Ended"
        return



    def switch_to_Main_Window(self):
        """ Main Controller Of Behavior For When Activating The MainWindow. """
        self.main_window.load_this_frame()
        self.main_window.file_label.configure(text="") 
        return
        
        
        
    def switch_to_Search_Window(self):
        """ Main Controller Of Behavior For When Activating The SearchWindow. """
        self.search_window.load_this_frame()
        return
        
        
        
    def switch_to_File_Window(self, index):
        """ Main Controller Of Behavior For When Activating The FileWindow. """
        self.file_window.load_this_frame(index % len(self.file_container))
        return
        
        
    
    def next_file(self):
        """ For When Next Button Is Pressed. """
        self.current_file_index = (self.current_file_index + 1) % len(self.file_container)
        self.switch_to_File_Window(self.current_file_index)
        return
        
        
        
    def previous_file(self):
        """ For When Previous Button Is Pressed. """
        self.current_file_index = (self.current_file_index - 1) % len(self.file_container)
        self.switch_to_File_Window(self.current_file_index)
        return