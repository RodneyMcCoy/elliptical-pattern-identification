"""
Rodney McCoy
rbmj2001@outlook.com
December 2022
"""

import os
from pathlib import Path






# A class for storing the relevant data for a single test flight. Stores,
#   - Relevent information to access test file
#   - Test file data
#   - Processed results (if the file has been processed)
class FileData:
    
    def __init__(self, name="", processed=False):
        self.name = name
        self.data = []
        self.processed_data = []
        self.is_processed = False 
        # Read in data into data
        
        
        
    # Returns just the file name, not the file extension or the file path to the given file
    def get_label(self):
        x = Path(self.name).name
        x = os.path.splitext(x)
        return x[0]
    
    
    
    # Returns raw data from file
    def get_data(self):
        return self.data
    
    
    
    # Returns processed data
    def get_processed_data(self):
        return self.processed_data