
import os, sys

filePath = os.path.abspath(__file__)
currentFolder = os.path.dirname(filePath)
sys.path.append(currentFolder)

from .MOTDet17Main import outputAsFramesToVideo