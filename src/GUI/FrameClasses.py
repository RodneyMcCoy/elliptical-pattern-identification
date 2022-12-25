"""
Rodney McCoy
rbmj2001@outlook.com
December 2022
"""

import tkinter as tk
import tkinter.ttk as ttk

import MainApp as App
import FileData as File






# Class for initialzing and controlling some behavior of the Sidebar Frame
class Sidebar(tk.Frame):
    
    def __init__(self, app):
        self.app_reference = app
        self.master = app.master
        self.frame = tk.Frame(self.master, relief=tk.RAISED, bd=2)



        # Create widgets for this frame
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
        

        
        # Place widgets onto the frame
        for index, key in enumerate(self.buttons):
            self.buttons[key].grid(row=index, column=0, sticky="ew", padx=5, pady=8)



        # Place frame onto the master window
        self.frame.grid(row=0, column=0, sticky="ns")
        return
    
    
    
    # Deactivate or activate all windows
    def state(self, activate : bool):
        if activate:
            for x in self.buttons:
                self.buttons[x].configure(state = tk.NORMAL)
        else:
            for x in self.buttons:
                self.buttons[x].configure(state = tk.DISABLED)
        return
    
        




# Class for initialzing and controlling some behavior of the Main Window Frame
class MainWindow(tk.Frame):
    
    def __init__(self, app):
        self.app_reference = app
        self.master = app.master
        self.frame = tk.Frame(self.master, relief=tk.RAISED, bd=0)



        # Create widgets for this frame
        self.file_button = ttk.Button(self.frame, text="Open a File",
            command = self.app_reference.addFile)
        self.folder_button = ttk.Button(self.frame, text="Open all Files in a Folder",
            command = self.app_reference.addFolder)
        self.file_label = ttk.Label(self.frame, text="Selected files will be placed here")
        self.process_button = ttk.Button(self.frame, text="Process Files", state = tk.DISABLED,
            command = self.app_reference.switch_to_Progress_Window)
        
        
        
        # Place widgets onto the frame
        self.file_button.pack()
        self.folder_button.pack()
        self.file_label.pack()
        self.process_button.pack()



        # Place frame onto the master window
        self.frame.grid(row=0, column=1, sticky="nsew")
        return
        
        

    # Purpose: Control behavior for when this frame is moved to the front
    def load_this_frame(self):
        self.frame.tkraise()
        return





# Class for initialzing and controlling some behavior of the Progress Window Frame
class ProgressWindow(tk.Frame):
    
    def __init__(self, app):
        self.app_reference = app
        self.master = app.master
        self.frame = tk.Frame(self.master)
        
        
        
        # Create Widgets for this frame
        self.label = ttk.Label(self.frame, text=
            "Definitely processing files. This might take a while (when the backend is actually done). Pressing any button on the left will stop the processing on the current file.")
        self.stop_button = ttk.Button(self.frame, text="Stop processing at the current file", 
            command = self.stop_processing)
        
        
        # Place widgets onto the frame
        self.label.pack()
        self.stop_button.pack()
        
        
        
        # Place the frame onto the master window
        self.frame.grid(row=0, column=1, sticky="nsew")
        return
        
        

    # Purpose: Stop processing the files
    def stop_processing(self):
        self.app_reference.sidebar.state(True)
        self.app_reference.switch_to_Main_Window()
        return
    
    
        
    # Purpose: Control behavior for when this frame is moved to the front
    def load_this_frame(self):
        self.frame.tkraise()
        self.app_reference.currently_processing = True
        return






# Class for initialzing and controlling some behavior of the Search Window Frame
class SearchWindow(tk.Frame):
    
    def __init__(self, app):
        self.app_reference = app
        self.master = app.master
        self.frame = tk.Frame(self.master)
        
        
        
        # Create widgets for this frame
        self.label = ttk.Label(self.frame, text="Definitely searching for files files.")
        self.options = []
        self.value = tk.StringVar()
        self.value.set( "Select From Files" )
        self.dropdown = ttk.Combobox(self.frame, value=self.options)
        self.submit_button = tk.Button(self.frame, text='Search for File', command= self.submit_answer)



        # Place widgets onto the frame
        self.label.pack()
        self.dropdown.pack()
        self.submit_button.pack()
        
        
        
        # Place the frame onto the master window
        self.frame.grid(row=0, column=1, sticky="nsew")
        return
    
        

    # Purpose: Control the behavior for when the submit button is pushed
    def submit_answer(self):
        answer = self.dropdown.get()
        
        if answer == "Select From Files" :
            self.app_reference.switch_to_File_Window(0)
        else:
            for i, val in enumerate(self.app_reference.file_container):
                if val.name == answer:
                    break
            self.app_reference.switch_to_File_Window(i)
            self.app_reference.current_file_index = i
        return



    # Purpose: Control behavior for when this frame is moved to the front
    def load_this_frame(self):
        # Refresh the search dropdown
        self.options = []
        for file in self.app_reference.file_container:
            self.options.append(file.name) 
                
        self.label.configure(text="".join(
            [" " + f.get_label() + " " for f in self.app_reference.file_container])) 
        
        self.dropdown.configure(value=self.options )

        # Move this frame to the front
        self.frame.tkraise()
        return






# Class for initialzing and controlling some behavior of the File Window Frame
class FileWindow(tk.Frame):
    
    def __init__(self, app):
        self.app_reference = app
        self.master = app.master
        self.frame = tk.Frame(self.master)
        
        
        # Create the widgets for this frame
        self.labels = []
        for i in range(5):
            self.labels.append(ttk.Label(self.frame, text="_"))
            
            
            
        # Place the widgets onto the frame
        for i in range(5):
            self.labels[i].grid(row=i, column=0, sticky="ew", padx=5, pady=4)
        
        
        
        # Place the frame onto the master window
        self.frame.grid(row=0, column=1, sticky="nsew")
        return
        
    

    # Purpose: Control behavior for when this frame is moved to the front
    def load_this_frame(self, index):
        # Refresh the top right file location indicators
        files = self.app_reference.file_container
        ind = []
        
        for i in range(-2, 2+1):
            if i != 0:
                ind.append((index + i) % len(files))
            elif i == 0:
                ind.append(index)
                
        
        for i in range(5):
            string = "(" + str(ind[i]) + ") "
            if i == 2:
                string += "CURRENT: "
            string += files[ind[i]].get_label()
            
            self.labels[i].configure(text=string) 
            
        # Place all relevant file data
        
        # MOve this frame to the front
        self.frame.tkraise()
        return