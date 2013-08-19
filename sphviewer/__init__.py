"""
Py-SPHViewer is an object-oriented rendering library. It was developed mainly
for rendering cosmological Smoothed Particles Hydrodynamical (SPH) simulations of galaxy formation, but in its current version, it can renderize any set of particles. 

Author: Alejandro Benitez-Llambay
E-mail: If you have any question, or you want to report bugs, issues, etc., please contact me at alejandrobll@oac.uncor.edu.
Acknowledgment: Many thanks to Pablo Benitez-Llambay. He has improved the original idea a lot, and without his help, Py-SPHViewer would not be what it is. 
"""

from Particles import Particles
from Camera import Camera
from Scene import Scene
from Render import Render
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    pos = np.random.rand(3,1000)
    mass = 1.*np.ones(1000)
    hue = np.random.rand(1000)
    sat = np.ones(1000)
    pos = np.random.rand(3,1000)
    P = Particles(pos[0,:], pos[1,:], pos[2,:], mass, prop1=hue,prop2=sat,nb=4) 
    S = Scene(P)
    I = Render.Fancy(S)
    print I.get_min(), I.get_max()
    I.histogram()
    plt.show()
    plt.imshow(I.get_image())
    plt.show()
    I.set_logscale()
    print I.get_min(), I.get_max()
    I.histogram()
    plt.show()
    plt.imshow(I.get_image())
    plt.show()

    I.set_logscale(False)
    print I.get_min(), I.get_max()
    I.histogram()
    plt.show()
    plt.imshow(I.get_image())
    plt.show()
