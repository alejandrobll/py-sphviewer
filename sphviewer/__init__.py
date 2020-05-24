"""
Py-SPHViewer is an object-oriented rendering library. It was developed mainly
for rendering cosmological Smoothed Particles Hydrodynamical (SPH) simulations of galaxy formation, but in its current version, it can renderize any set of particles. 

Author: Alejandro Benitez-Llambay
E-mail: If you have any question, or you want to report bugs, issues, etc., please contact me at bllalejandro@gmail.com
Acknowledgment: Many thanks to Pablo Benitez-Llambay. He has improved the original idea a lot, and without his help, Py-SPHViewer would not be what it is. 
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt

from .Particles import Particles
from .Camera import Camera
from .Scene import Scene
from .Render import Render
from .version import __version__
