"""
Rodney McCoy
rbmj2001@outlook.com
December 2022
"""

import os
from pathlib import Path

class FileData:
    
    def __init__(self, name="", processed=False):
        self.name = name
        self.data = []
        self.is_processed = False 
        # Read in data into data
    
    def get_label(self):
        x = Path(self.name).name
        x = os.path.splitext(x)
        return x[0]
    
    def get_data(self):
        return self.data
        