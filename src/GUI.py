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



# %% ----- Function Defs -----



# Purpose: Close window and python interpreter
def close(self):
    # BUG weird errors thrown when code ran
    self.destroy()
    sys.exit()



# Purpose: determine if input satisfies proper formating
def formatted_properly(filename) -> bool:
    split_tup = os.path.splitext('my_file.txt')
    if split_tup[1] != ".csv":
        return False
    # TODO Not finished
    return False
    
    
    
# Purpose: Add a single file to "files" list
def addFile():
    # Get file from user
    filename = tk.filedialog.askopenfilename(
        initialdir = "./", title = "Select a File", filetypes = 
        (("Text files", "*.csv*"), ("all files", "*.*")))
     
    # Add File to "files" list
    files.append(filename)
    # TODO ignore input and notify user if a file that is already in files is inserted
    # TODO ignore input and notify user if a file doesnt  follow format

    # Update GUI with new files
    file_label.configure(text="".join(
        [" " + os.path.basename(f)  for f in files])) 
    


# Purpose: Add all files in folder to "files" list
def addFolder():
    # Get file from user
    foldername = tk.filedialog.askdirectory(
        initialdir = "./", title = "Select a Folder")
     
    # Add File to "files" list
    files.append(foldername)
    # TODO iterate through folder, only adding files if they satisfy formatting
    
    # Update GUI with new files
    file_label.configure(text="".join(
        [" " + os.path.basename(f) + " " for f in files])) 



# %% ----- Tkinter Code -----



# create the application window
root = tk.Tk()

# assign title to the window
root.title("Ellipitical Pattern Identification")

# control intial size of window
root.state('zoomed')
root.geometry("800x500")

# set app icon
path = Path(os.getcwd()).parent.absolute() / "res" / "logo.ico"
root.iconbitmap(path)

# File Holder
files = []



# configure rows and cols
root.rowconfigure(0, minsize=800, weight=1)
root.columnconfigure(1, minsize=800, weight=1)

# split tkinter window into left and right. left for buttons, right for everything else
right_frame = tk.Frame(root, relief=tk.RAISED, bd=0)
left_frame = tk.Frame(root, relief=tk.RAISED, bd=2)

# initialize widgets on left frame
close_button = ttk.Button(left_frame, text="Quit", 
                          command = lambda root=root:close(root))
file_button = ttk.Button(left_frame, text="Open a File",
                          command = addFile)
folder_button = ttk.Button(left_frame, text="Open all Files in a Folder",
                          command = addFolder)

# initialize widgets on right frame
file_label = ttk.Label(right_frame, text="Selected files will be placed here")


# place widgets on left frame
close_button.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
file_button.grid(row=1, column=0, sticky="ew", padx=5)
folder_button.grid(row=2, column=0, sticky="ew", padx=5)

# place widgets on right frame
file_label.pack()


# place left and right frame onto root (window)
left_frame.grid(row=0, column=0, sticky="ns")
right_frame.grid(row=0, column=1, sticky="nsew")



# Start The Main Tkinter Loop
root.mainloop()

