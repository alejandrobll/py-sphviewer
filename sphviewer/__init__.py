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
    import time
    n1 = 10000 #number of particles to make the disk
    n2 = n1/3  # number of particles to make the background
    r  = np.random.rand(n1)
    phi = np.pi*np.random.rand(n1)
    pos = np.zeros([3,n1], dtype=np.float32)
    pos[0,:] = r*np.cos(phi)
    pos[1,:] = r*np.sin(phi)
    pos[2,:] = 0.1*np.random.rand(n1)
    
    background = -2+4*np.random.rand(3,n2)
    pos = np.concatenate((pos,background),axis=1)
    
    mass = np.ones(n1+n2)

    Particles1 = Particles(pos,mass,sort=True)
    Scene1 = Scene(Particles1)
    Render1 = Render(Scene1)
    for i in xrange(100):
        start = time.time()
        Scene1.update_camera(r=2.00, t=90, roll=30, xsize=1920, ysize=1080)
        Render1 = Render(Scene1)
        Render1.set_logscale()
        img = Render1.get_image()
        extent = Render1.get_extent()
        stop = time.time()
        print 'Time = ', stop-start
        fig = plt.figure(1,figsize=(5,5))
        ax1 = fig.add_subplot(111)
        ax1.imshow(img, extent=extent, origin='lower', cmap='hot')
        ax1.set_xlabel('X', size=15)
        ax1.set_ylabel('Y', size=15)
        
        plt.show()
