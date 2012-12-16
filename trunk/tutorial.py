import matplotlib.pyplot as plt
import numpy as np
import sphviewer

"""
Tutorial of sphviewer.
"""

#Generating data
pos = np.random.rand(3,1000)

#Generating the smoothed data
image = sphviewer.scene(pos=pos, nb=4)
dens, bins, extent = image.make_scene()

#Ploting
plt.imshow(np.log10(dens+1.), 
           origin='lower', 
           interpolation='nearest', 
           cmap='gray')
plt.show()
