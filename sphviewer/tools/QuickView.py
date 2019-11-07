# This tool is part of sphviewer. It is intended to be used as a simple
# way to get quick images of the simulations.
# Author: Alejandro Benitez-Llambay

from __future__ import absolute_import, division, print_function

import sphviewer as sph
import matplotlib.pyplot as plt
import numpy as np


class QuickView(object):
    def __init__(self, pos, mass=None, hsml=None, nb=None,
                 logscale=True, plot=True, min_hsml=None,
                 max_hsml=None, **kwargs):

        if(mass is None):
            mass = np.ones(len(pos))

        if(nb == None):
            self._P = sph.Particles(pos, mass, hsml)
        else:
            self._P = sph.Particles(pos, mass, hsml, nb)

        if((min_hsml is not None) or (max_hsml is not None)):
            hsml = self.get_hsml()
            if(min_hsml is not None):
                min_hsml = min_hsml
            else:
                min_hsml = np.min(hsml)
            if(max_hsml is not None):
                max_hsml = max_hsml
            else:
                max_hsml = np.max(hsml)

            hsml = np.clip(hsml, min_hsml, max_hsml)
            print('Limiting smoothing length to the range '
                  '[%.3f,%.3f]' % (min_hsml, max_hsml))
            self._P.set_hsml(hsml)

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
            print('Image saved in', filename)
        except:
            print('Error while saving image')
        return

    def get_hsml(self):
        return self._P.get_hsml()


if __name__ == '__main__':
    import h5py
    halo = h5py.File('../../examples/dm_halo.h5py', 'r')
    pos = halo['Coordinates'].value

    qv = QuickView(pos, r='infinity', nb=8)
