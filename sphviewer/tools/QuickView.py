#This tool is part of sphviewer. It is intended to be used as a simple
#way to get quick images of the simulations.
#Author: Alejandro Benitez-Llambay

import sphviewer as sph
import matplotlib.pyplot as plt
import numpy as np

class QuickView():
    def    __init__(self, pos, mass=None, hsml=None, nb=None,
                    logscale=True, plot=True, **kwargs):

        if(mass is None):
            mass = np.ones(len(pos[0,:]))
                    
        if(nb == None):
            self._P = sph.Particles(pos, mass, hsml)
        else:
            self._P = sph.Particles(pos, mass, hsml, nb)

        self._S = sph.Scene(self._P)
        self._S.update_camera(**kwargs)

        self._R = sph.Render(self._S)
        if(logscale):
            self._R.set_logscale()

        self._img = self._R.get_image()
        self._extent = self._R.get_extent()

        if(plot):
            self.imshow(aspect='auto')
            return
        else:
            return
            
    def imshow(self, **kwargs):
        ax = plt.gca()
        ax.imshow(self._img, extent=self._extent, origin='lower', **kwargs)
        plt.show()

    def get_image(self):
        return self._img

    def get_extent(self):
        return self._extent

    def imsave(self, filename, **kwargs):
        try:
            plt.imsave(filename, self._img, **kwargs)
            print 'Image saved in '+ filename
        except:
            print 'Error while saving image'
        return
        
        
    
