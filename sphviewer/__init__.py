from Particles import Particles
from Camera import Camera
from Scene import Scene
import Render
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    pos = np.random.rand(3,1000)
    mass = 1.*np.ones(1000)
    hue = np.random.rand(1000)
    sat = np.ones(1000)
    pos = np.random.rand(3,1000)
    P = Particles(pos[0,:], pos[1,:], pos[2,:], mass, prop1=hue,prop2=sat)
    S = Scene(P)
    I = Render.Fancy(S)
    print I.get_min(), I.get_max()
    I.histogram()
    plt.show()
    plt.imshow(I.get_image())
    plt.show()
