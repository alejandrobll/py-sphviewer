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
import matplotlib.pyplot as plt


# Tools for blending images. There are plenty of possibilities, and
# depending on the images, the final result can vary significantly.
class Blend(object):
    def __init__(self, image1, image2):
        self.image1 = image1
        self.image2 = image2

    def Screen(self):
        output = np.zeros(np.shape(self.image1))

        if np.shape(self.image1)[2] == 4:  # test for RGBA input image
            output[:, :, 3] = 1  # set alpha values to unity

        for i in range(3):
            output[:, :, i] = (1.0-(1.0-self.image1[:, :, i]) *
                               (1.0-self.image2[:, :, i]))
        return output

    def Overlay(self):
        output = np.zeros(np.shape(self.image1))

        if np.shape(self.image1)[2] == 4:  # test for RGBA input image
            output[:, :, 3] = 1  # set alpha values to unity

        for i in range(3):
            output[:, :, i] = (self.image1[:, :, i] *
                               (self.image1[:, :, i]+2*self.image2[:, :, i] *
                                (1.0-self.image1[:, :, i])))

        return output


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from sphviewer.tools import QuickView
    from matplotlib import cm

    def get_normalized_image(image):
        image = (image-np.min(image))/(np.max(image)-np.min(image))
        return image

    pos1 = np.zeros([5000, 3])
    pos2 = np.zeros([10000, 3])

    pos1[:, 0:2] = -1+2*np.random.rand(5000, 2)
    pos2[:, 0:2] = -1+2*np.random.rand(10000, 2)

    r2 = np.sqrt(pos2[:, 0]**2+pos2[:, 1]**2)
    k2, = np.where(r2 < 0.5)
    pos2 = pos2[k2, :]

    qv1 = QuickView(pos1.T, np.ones(len(pos1)),
                    r='infinity', logscale=False, plot=False,
                    extent=[-1, 1, -1, 1], x=0, y=0, z=0)
    qv2 = QuickView(pos2.T, np.ones(len(pos2)),
                    r='infinity', logscale=False, plot=False,
                    extent=[-1, 1, -1, 1], x=0, y=0, z=0)

    image1 = cm.gist_heat(get_normalized_image(qv1.get_image()))
    image2 = cm.gist_stern(get_normalized_image(qv2.get_image()))

    fig = plt.figure(1, figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    blend = Blend(image1, 1.0*image2)
    screen = blend.Screen()
    overlay = blend.Overlay()

    ax1.imshow(screen, origin='lower', extent=qv1.get_extent())
    ax1.set_title('Screen')
    ax2.imshow(overlay, origin='lower', extent=qv1.get_extent())
    ax2.set_title('Overlay')
    plt.show()
