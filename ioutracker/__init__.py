__version__ = "1.1.1"

import os, sys

filePath = os.path.abspath(__file__)
currentFolder = os.path.dirname(filePath)
sys.path.append(currentFolder)

from .dataloaders import *
from .src import *
from .metrics import *
from .inference import *
