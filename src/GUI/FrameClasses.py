"""
Rodney McCoy
rbmj2001@outlook.com
December 2022
"""

import tkinter as tk
import tkinter.ttk as ttk

import MainApp as App
import FileData as File


# %% ----- Separate Class for each Frame -----


class Sidebar(tk.Frame):
    
    def __init__(self, app):
        self.app_reference = app
        self.master = app.master
        self.frame = tk.Frame(self.master, relief=tk.RAISED, bd=2)


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
        

        
        

        for index, key in enumerate(self.buttons):
            self.buttons[key].grid(row=index, column=0, sticky="ew", padx=5, pady=8)


        self.frame.grid(row=0, column=0, sticky="ns")
        
        
        
    def close_windows(self):
        self.master.destroy()
        
    def activate_buttons(self):
        self.buttons["next"].configure(state = tk.NORMAL)
        self.buttons["previous"].configure(state = tk.NORMAL)
        self.buttons["select"].configure(state = tk.NORMAL)

        
     





class MainWindow(tk.Frame):
    
    def __init__(self, app):
        self.app_reference = app
        self.master = app.master
        self.frame = tk.Frame(self.master, relief=tk.RAISED, bd=0)

        self.file_button = ttk.Button(self.frame, text="Open a File",
                                  command = self.app_reference.addFile)
        self.folder_button = ttk.Button(self.frame, text="Open all Files in a Folder",
                                  command = self.app_reference.addFolder)

        self.file_label = ttk.Label(self.frame, text="Selected files will be placed here")
        
        self.process_button = ttk.Button(self.frame, text="Process Files", state = tk.DISABLED,
                                  command = self.app_reference.switch_to_Progress_Window)
        
        self.file_button.pack()
        self.folder_button.pack()
        self.file_label.pack()
        self.process_button.pack()

        self.frame.grid(row=0, column=1, sticky="nsew")
        
        
     
    def close_windows(self):
        self.master.destroy()



    def load_this_frame(self):
        self.frame.tkraise()


    def activate_buttons(self):
        self.process_button.configure(state = tk.NORMAL)






class ProgressWindow(tk.Frame):
    
    def __init__(self, app):
        self.master = app.master
        self.frame = tk.Frame(self.master)
        self.label = ttk.Label(self.frame, text=
            "Definitely processing files. This might take a while (when the backend is actually done)")
        self.label.pack()
        self.frame.grid(row=0, column=1, sticky="nsew")
        
        
        
    def close_windows(self):
        self.master.destroy()



    def load_this_frame(self):
        # TODO BEHAVIOR FOR APPLYING BACKEND ALGORITHMS TO FILES GOES HERE
        self.frame.tkraise()






class SearchWindow(tk.Frame):
    
    def __init__(self, app):
        self.app_reference = app
        self.master = app.master
        self.frame = tk.Frame(self.master)
        self.label = ttk.Label(self.frame, text="Definitely searching for files files.")

        
        
        self.options = []

        self.value = tk.StringVar()

        self.value.set( "Select From Files" )
        
        # Create Dropdown menu
        self.dropdown = ttk.Combobox(self.frame, value=self.options)
        
        self.submit_button = tk.Button(self.frame, text='Search for File', command= self.submit_answer)

        self.label.pack()
        
        self.dropdown.pack()
            
        self.submit_button.pack()
        
        
        self.frame.grid(row=0, column=1, sticky="nsew")
        # self.frame.pack(fill = tk.BOTH, expand = True)
        

    def submit_answer(self):
        answer = self.dropdown.get()
        if answer == "Select From Files" :
            self.app_reference.switch_to_File_Window(0)
        else:
            for i, val in enumerate(self.app_reference.file_container):
                if val.name == answer:
                    break
            self.app_reference.switch_to_File_Window(i)


        
        
    def close_windows(self):
        self.master.destroy()



    def load_this_frame(self):
        if self.app_reference.file_container == []:
            self.label.configure(text="No Files Have Been Inputted")
        else:
            self.options = []
            for file in self.app_reference.file_container:
                self.options.append(file.name) 
                    
            self.label.configure(text="".join(
                [" " + f.get_label() + " " for f in self.app_reference.file_container])) 
            
            self.dropdown.configure(value=self.options )


        self.frame.tkraise()






class FileWindow(tk.Frame):
    
    def __init__(self, app):
        self.app_reference = app
        self.master = app.master
        self.frame = tk.Frame(self.master)
        
        
        self.labels = []
        
        for i in range(5):
            self.labels.append(ttk.Label(self.frame, text="_"))
            
        for i in range(5):
            self.labels[i].grid(row=i, column=0, sticky="ew", padx=5, pady=4)
        
        self.frame.grid(row=0, column=1, sticky="nsew")
        

        
    def close_windows(self):
        self.master.destroy()



    def load_this_frame(self, index):
        # TODO Behavior for showing file data goes here
        self.frame.tkraise()
        
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