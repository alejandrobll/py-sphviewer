#This file is part of Py-SPHViewer

#<Py-SPHVIewer is a framework for rendering particles in Python
#using the SPH interpolation scheme.>
#Copyright (C) <2013>  <Alejandro Benitez Llambay>

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import absolute_import, division, print_function

import sphviewer as sph
import matplotlib.pyplot as plt
import numpy as np


class QuickView(object):
    def __init__(self, pos, mass=None, hsml=None, nb=None,
                 logscale=True, plot=True, min_hsml=None,
                 max_hsml=None, **kwargs):

        """
        Quickview is a simple wrapper of the sphviewer API. 
        This utility stores the particles, defines the Scene and the Camera, and produces the rendering of the Scene in one step, thus making the process of producing images of SPH particles easy and quick. 
        The arguments of the functions are:
         - pos = SPH particle positions. Array of shape [N,3], where N in the 
           number of particles and pos[:,0] = x; pos[:,1] = y; pos[:,2] = z.
         - mass (optional): is the mass of the SPH particles. If present, 
           it must be an array of size N. If absent, a constant value, 
           mass = 1, will be assumed for all particles.
         - hsml (optional): this is an array containing the smoothing lenghts 
           of the SPH particles. The absence of the array will trigger the 
           calculation of the smoothing lenghts.
         - nb (optional): number of neighbours to be used for the calculation
           of the SPH smoothing lengths. If absent, the default value nb=32 will
           be used. This arguement is ignored if the array hsml is provided.
         - logscale (optional): If True, the output image becomes 

                out_log = log10(1.0+output)

         - plot (optional): If true, QuickView will plot the resulting image in
           the current active figure.
         - min_hsml / max_hsml: Physical values of the minimum / maximum 
           smoothing lengths. Only used when defined. 
         
         **kwargs 
         These include the parameters of the Camera:

        """

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
