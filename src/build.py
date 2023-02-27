"""
Rodney McCoy
rbmj2001@outlook.com
Feb 2023
"""

# This script executes the python code to build the pyinstaller application

import os
from pathlib import Path
import PyInstaller.__main__



MAIN_PY = Path(os.getcwd()).absolute() / "GUI" / "main.py"

PyInstaller.__main__.run([
    str(MAIN_PY),
    '-n EllipticalPatternIdentification',
    '--onefile',
])