#//////////////////////////////////////////////////////
#This tools is part of py-sphviewer. 
#Author: Alejandro Benitez-Llambay
#//////////////////////////////////////////////////////
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
