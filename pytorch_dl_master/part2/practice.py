import sys, os
sys.path.append(os.getcwd())
from functions.util import irc2xyz, xyz2irc
import numpy as np
import math

coord_xyz_test = np.random.randn(3)
origin_xyz_test = np.random.normal(loc=20, scale=100)

direction_test = np.array([[1, 0, 0],
                           [0, np.cos(math.pi/3), -np.sin(math.pi/3)],
                           [0, np.sin(math.pi/3), np.cos(math.pi/3)]])


