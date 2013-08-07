from Particles import Particles
from Camera import Camera
from Scene import Scene
from Render import Fancy
from Render import Render
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    pos = np.random.rand(3,1000)
    mass = 40.*np.ones(1000)
    hue = np.random.rand(1000)
    sat = np.ones(1000)
    pos = np.random.rand(3,1000)
    P = Particles(pos[0,:], pos[1,:], pos[2,:], mass)
    S = Scene(P)
    I = Render(S)
    plt.imshow(I.get_image())
    plt.show()
