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
        # Denots Whether Program Is Currently Processing Files.
        self.currently_processing = False
                
        # Initialize Each Class / Frame In "FrameClasses.py".
        self.progress_window = Frame.ProgressWindow(self)
        self.search_window = Frame.SearchWindow(self)
        self.main_window = Frame.MainWindow(self)
        self.sidebar = Frame.Sidebar(self)
        self.file_window = Frame.FileWindow(self)

        # Ensure The Starting Class / Frame Is MainWindow.
        self.switch_to_Main_Window()
        return

    

    def close(self):
        """ Controls When Tkiner WM_DELETE_WINDOW Is Raised. Meant To Cleanly
        Exit Python Interpreter. """
        if self.currently_processing:
            self.progress_window.stop_processing()
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
            self.main_window.process_button.configure(state = tk.NORMAL)
            self.sidebar.buttons["next"].configure(state = tk.NORMAL)
            self.sidebar.buttons["previous"].configure(state = tk.NORMAL)
            self.sidebar.buttons["select"].configure(state = tk.NORMAL)


        # Add The Inputted File To "file_container" If Its Not Already In Tt
        add_file = True
        for file in self.file_container:
            if file == filename:
                add_file = False
        if add_file:
            self.file_container.append(filename)

        # Update MainWindow With Info About New File.
        self.main_window.file_label.configure(text="".join(
            [" " + f + "\n " for f in self.file_container])) 
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
        
        FILES = os.listdir(foldername)
        
        # If This Is The First File Inputted, Activate Buttons Which Assume
        # Files Are Already Given.      
        if self.file_container == []:
            self.main_window.process_button.configure(state = tk.NORMAL)
            self.sidebar.buttons["next"].configure(state = tk.NORMAL)
            self.sidebar.buttons["previous"].configure(state = tk.NORMAL)
            self.sidebar.buttons["select"].configure(state = tk.NORMAL)
                
            
        FoundProperFile = False
        for filename in FILES:
            # Add The Inputted Files To "file_container" If Its Not Already In It
            if os.path.splitext(filename)[-1].lower() == ".txt":
                FoundProperFile = True
                add_file = True
                for file in self.file_container:
                    if file == filename:
                        add_file = False
                if add_file:
                    this_path = Path(foldername) / Path (filename)
                    self.file_container.append(str(this_path))
            
        if FoundProperFile == False:
            self.main_window.file_label.configure(text="No file of proper formatting (ending in .txt) was found in folder.")
            return


        
        # Update MainWindow With Info About New Files.
        self.main_window.file_label.configure(text="".join(
            [" " + f + "\n " for f in self.file_container])) 
        return



    def switch_to_Progress_Window(self):
        """ Main Controller Of Behavior For When Activating The ProgressWindow """
        # Load Progress Window.
        self.progress_window.load_this_frame()
        
        # Freeze Buttons In Sidebar Window.
        self.sidebar.state(False)
        
        # Create Secondary Process For Using Backend.

        self.backend, frontend = multiprocessing.Pipe(duplex = True)
        self.process = multiprocessing.Process(target=BackEndInterface.OpenFileProcessingThread,
                                               args= (frontend, self.file_container))

        self.is_processing_done()
        self.process.start()

        return



    def is_processing_done(self):
        if not self.currently_processing:
            return
        if self.backend.poll():
            pipe = self.backend.recv()
            if pipe == "STOP":
                self.stop_processing()
                return
            print(pipe)
            self.progress_window.label["text"] = pipe
        
        self.master.after(250, self.is_processing_done)
        return


        
    def stop_processing(self):
        """ For When Processing Needs To Stop. Deactivates / Deletes Secondary
        Process. """
        self.sidebar.state(True)
        self.switch_to_Main_Window()
        self.currently_processing = False
        self.backend.send("STOP")
        # self.pipe_to_backend.close()
        return



    def switch_to_Main_Window(self):
        """ Main Controller Of Behavior For When Activating The MainWindow. """
        self.main_window.load_this_frame()
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