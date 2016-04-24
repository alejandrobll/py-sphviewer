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
            P = sph.Particles(pos, mass, hsml)
        else:
            P = sph.Particles(pos, mass, hsml, nb)

        S = sph.Scene(P)
        S.update_camera(**kwargs)

        R = sph.Render(S)
        if(logscale):
            R.set_logscale()

        self._img = R.get_image()
        self._extent = R.get_extent()

        if(plot):
            self.imshow(aspect='auto')
            return
        else:
            return
            
    def imshow(self, **kwargs):
        ax = plt.gca()
        ax.imshow(self._img, extent=self._extent, **kwargs)
        plt.show()

    def get_image(self):
        return self.img

    def get_extent(self):
        return self.extent

    def imsave(self, filename, **kwargs):
        try:
            plt.imsave(filename, self._img, **kwargs)
            print 'Image saved in '+ filename
        except:
            print 'Error while saving image'
        return
        
    
