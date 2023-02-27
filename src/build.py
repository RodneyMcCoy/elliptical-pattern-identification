"""
Rodney McCoy
rbmj2001@outlook.com
Feb 2023
"""

# This script executes the python code to build the pyinstaller application

import os
from pathlib import Path


MAIN_PY = Path(os.getcwd()) / "GUI" / "main.py"

BUILD_FOLDER = Path(os.getcwd()).parent.absolute()


import PyInstaller.__main__

PyInstaller.__main__.run([
    str(MAIN_PY),
    '-n EllipticalPatternIdentification',
    '--onefile',
    '--specpath ' + str(BUILD_FOLDER)
])