# This file is part of Py-SPHViewer

# <Py-SPHVIewer is a framework for rendering particles in Python
# using the SPH interpolation scheme.>
# Copyright (C) <2013>  <Alejandro Benitez Llambay>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pylab as plt

class Camera(object):
    def __init__(self, x=None, y=None, z=None, r=None,
                 t=None, p=None, zoom=None, roll=None,
                 xsize=None, ysize=None, extent=None):

        """
        The Camera class is a container that stores the camera parameters. The camera is an object that lives in the space and has spherical coordinates (r,theta,phi), centred around the location (x,y,z). Angles *theta* and *phi* are given in degrees, and enable to rotate the camera along the x-axis and y-axis, respectively. The *roll* angle induces rotations along the line-of-sight, i.e., the z-axis. 

        xsize and ysize correspond to the size of the image, in pixels, that the Camera will produce once the Render class is used.

        By default, the Camera will see the space using a perspective projection, with field of view defined by the zoom parameter. The relation between the field of view, FoV, and the zoom, M, is: 

            tan(FoF/2) = 1/M.

        Thus, the default zoom value of 1 will return the default
        field of view of 90 degrees. 
    
        If the value of *r* is r='infinity', the perspective projection is
        replaced by a parallel projection, which corresponds to a camera placed 
        at the infinity. In this case, the field of view does not have any 
        meaning, and it is replaced by the *extent* argument, which defines the 
        field of view in linear (rather than angular) units, relative to the 
        centre of the camera, (x,y). The *extent* argument must be an 
        array of four elements, e.g., extent=[-l,l,-l,l], in which each 
        entry indicates the extent of the final image relative to the camera centre. For this particular example, the Camera will see everything between x-l, x+l in the x-direction and everything between y-l and y+l in the y direction. 
        
        Special care must be taken when combining *extent* with the number of pixels of image, i.e., *xsize* and *ysize*, as this can impact the aspect ratio of the pixels.
        """

        self._name = 'CAMERA'
        self.__params = {'x': x, 'y': y, 'z': z, 'r': r,
                         't': t, 'p': p, 'zoom': zoom, 'roll': roll,
                         'xsize': xsize, 'ysize': ysize, 'extent': extent}

    def get_params(self):
        """
        Use this function to get the parameters of the camera.
        """
        return self.__params

    def set_params(self, **kargs):
        """
        Use this function set any parameter of the camera. This is useful
        to avoid instantiating the entire Camera class from scratch. 
        Valid parameters are: 
         - set_params(
             x, y, z, 
             r, t, p, roll,zoom
             xsize=None, ysize=None, 
             extent=None)
        """
        for key in kargs:
            self.__params[key] = kargs[key]

    def _get_camera(self, plane, **kargs):
        x0 = -self.__params['r']*(np.sin(self.__params['p']*np.pi/180.) *
                                  np.cos(self.__params['t']*np.pi/180.))
        z0 = self.__params['r']*(np.cos(self.__params['p']*np.pi/180.) *
                                 np.cos(self.__params['t']*np.pi/180.))
        y0 = self.__params['r']*np.sin(self.__params['t']*np.pi/180.)

        xcam = self.__params['x']+x0
        ycam = self.__params['y']+y0
        zcam = self.__params['z']+z0

        if(plane == 'xy'):
            camera = plt.Line2D([xcam], [ycam], c='m',
                                marker='o', markersize=15, **kargs)
            arrow = plt.Line2D([xcam, self.__params['x']], [
                               ycam, self.__params['y']], lw=5)
        elif(plane == 'xz'):
            camera = plt.Line2D([xcam], [zcam], c='m',
                                marker='o', markersize=15, **kargs)
            arrow = plt.Line2D([xcam, self.__params['x']], [
                               zcam, self.__params['z']], lw=5)
        elif(plane == 'yz'):
            camera = plt.Line2D([ycam], [zcam], c='m',
                                marker='o', markersize=15, **kargs)
            arrow = plt.Line2D([ycam, self.__params['y']], [
                               zcam, self.__params['z']], lw=5)
        else:
            print('Incorrect plane:', plane)
            print("Possibles planes are: 'xy';'xz';'yz'")
            return
        return camera, arrow

    def plot(self, plane, axis=None, **kargs):
        if(axis == None):
            axis = plt.gca()
        camera, arrow = self._get_camera(plane)
        axis.add_line(camera, **kargs)
        axis.add_line(arrow, **kargs)

    def set_autocamera(self, Particles, mode='minmax'):
        """
        Use this function to obtain a good guess of the camera parameters
        for the current distribution of particles sotored in the 
        Particles class. Setting the camera of an uknown distribution of 
        particles may be tricky. This function should help with this.
        Current autocamera modes are: 
            - 'minmax', 'median', 'mean', and 'density'.
        """
        try:
            particles_name = Particles._name
        except AttributeError:
            print("You must use a valid class...")
            return
        if (particles_name != 'PARTICLES'):
            print("You must use a valid Particles class...")
            return
        xmax, ymax, zmax = (np.max(Particles._pos[:, 0]),
                            np.max(Particles._pos[:, 1]),
                            np.max(Particles._pos[:, 2]))
        xmin, ymin, zmin = (np.min(Particles._pos[:, 0]),
                            np.min(Particles._pos[:, 1]),
                            np.min(Particles._pos[:, 2]))

        if(mode == 'minmax'):
            xmean = (xmax+xmin)/2.
            ymean = (ymax+ymin)/2.
            zmean = (zmax+zmin)/2.

        if(mode == 'density'):
            k = np.argmin(Particles.get_hsml()[:])
            xmean = Particles._pos[k, 0]
            ymean = Particles._pos[k, 1]
            zmean = Particles._pos[k, 2]

        if(mode == 'median'):
            xmean = np.median(Particles._pos[:, 0])
            ymean = np.median(Particles._pos[:, 1])
            zmean = np.median(Particles._pos[:, 2])

        if(mode == 'mean'):
            xmean = np.mean(Particles.get_pos()[:, 0])
            ymean = np.mean(Particles.get_pos()[:, 1])
            zmean = np.mean(Particles.get_pos()[:, 2])

        r = np.sqrt((xmax-xmin)**2+(ymax-ymin)**2+(zmax-zmin)**2)

        self.__params = {'x': xmean, 'y': ymean, 'z': zmean, 'r': r,
                         't': 0, 'p': 0, 'zoom': 1, 'roll': 0,
                         'xsize': 500, 'ysize': 500, 'extent': None}
