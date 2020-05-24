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

import numpy as np

from .makehsv import makehsv


def image_from_hsv(h = 0, v = 0, 
                   img_hmin = None, img_hmax=None,
                   img_vmin = None, img_vmax=None,
                   hmin = None, hmax=None):

    if(img_hmin == None): img_hmin = np.min(h)
    if(img_hmax == None): img_hmax = np.max(h)
    if(img_vmin == None): img_vmin = np.min(v)
    if(img_vmax == None): img_vmax = np.max(v)
    if(hmin  == None): hmin = 0.0
    if(hmax  == None): hmax = 1.0

    ysize = np.shape(v)[1]
    xsize = np.shape(v)[0]

    image = np.zeros([ysize,xsize,3], dtype=np.float32)

    r, g, b= makehsv(h, v, img_hmin, img_hmax,
                     img_vmin, img_vmax,
                     hmin, hmax)

    image[:,:,0] = np.reshape(r, [ysize,xsize])
    image[:,:,1] = np.reshape(g, [ysize,xsize])
    image[:,:,2] = np.reshape(b, [ysize,xsize])

    return image
