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

# from scipy import weave
# from scipy.weave import converters
import numpy as np
import matplotlib.pyplot as plt
import os
import colorsys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def import_code(filename):
    # naive function to import c code
    string = ''
    fi = open(filename, 'r').readlines()
    for i in fi:
        string += i
    return string


class Render(object):
    def __init__(self, Scene):
        """
        Produces a render of the Scene. It uses a kernel interpolation 
        method. Its setting and getting methods are:

        getting methods:
        ----------------

        - get_image
        - get_min
        - get_max
        - get_extent
        - get_logscale

        setting methods:
        ----------------

        - set_logscale

        Other methods are:

        - save 
        - histogram

        """
        try:
            class_name = Scene._name
        except AttributeError:
            print("ERROR: use a valid class...")
            return
        if class_name != "SCENE":
            print("ERROR: use a valid class...")
            return

        self.Scene = Scene

        xsize = Scene.Camera.get_params()["xsize"]
        ysize = Scene.Camera.get_params()["ysize"]

        # We need to correct the image by 1 / pixel area
        # as all calculations are done in grid pixel units.

        extent = Scene.get_extent()
        x_extent = extent[1] - extent[0]
        y_extent = extent[3] - extent[2]

        pixel_area = (x_extent / xsize) * (y_extent / ysize)

        self.__image = self.__make_render(Scene._x, Scene._y, Scene._hsml,
                                          Scene._kview, xsize, ysize) / pixel_area

        # lets define some flags
        self.__logscale_flag = False

    def __make_render(self, x, y, t, kview, xsize, ysize):
        from .extensions import render

        image = render.render(
            self.Scene._x,
            self.Scene._y,
            self.Scene._hsml,
            self.Scene._m,
            np.int32(xsize),
            np.int32(ysize),
        )
        return np.reshape(image, [ysize, xsize])

    def get_image(self):
        """
        - get_image(): Returns the image, with dimension [xsize,ysize], 
        where xsize and ysize are the number of pixels of the image defined by the Camera.
        """
        extent = self.Scene.get_extent()
        return self.__image

    def get_max(self):
        """
        - get_max(): Returns the maximum value of the image.
        """
        return np.max(self.__image)

    def get_min(self):
        """
        - get_min(): Returns the minimum value of the image.
        """
        return np.min(self.__image)

    def get_extent(self):
        """
        - get_extent(): call to sphviewer.Scene.get_extent().
        """
        return self.Scene.get_extent()

    def set_logscale(self, t=True):
        """
        - set_logscale(): If M is the matrix of the image, it defines the image M as log10(M+1).
        """
        if t == self.get_logscale():
            return
        else:
            if(t):
                self.__image = np.log10(self.__image+1)
                self.__logscale_flag = True
            else:
                self.__image = 10**self.__image-1.
                self.__logscale_flag = False

    def get_logscale(self):
        """
        - get_logscale(): Returns True or False depending on the scale of the image.
        """
        return self.__logscale_flag

    def histogram(self, axis=None, **kargs):
        """
        - histogram(axis=None, **kargs): It computes and shows the histogram of the image. This is 
        usefull for choosing a proper scale to the output, or for clipping some values. If 
        axis is None, it selects the current axis to plot the histogram.

        Keyword arguments:

        *bins*:
        Either an integer number of bins or a sequence giving the
        bins.  If *bins* is an integer, *bins* + 1 bin edges
        will be returned, consistent with :func:`numpy.histogram`
        for numpy version >= 1.3, and with the *new* = True argument
        in earlier versions.
        Unequally spaced bins are supported if *bins* is a sequence.

        *range*:
        The lower and upper range of the bins. Lower and upper outliers
        are ignored. If not provided, *range* is (x.min(), x.max()).
        Range has no effect if *bins* is a sequence.

        If *bins* is a sequence or *range* is specified, autoscaling
        is based on the specified bin range instead of the
        range of x.

        *normed*:
        If *True*, the first element of the return tuple will
        be the counts normalized to form a probability density, i.e.,
        ``n/(len(x)*dbin)``.  In a probability density, the integral of
        the histogram should be 1; you can verify that with a
        trapezoidal integration of the probability density function::

        pdf, bins, patches = ax.hist(...)
        print(np.sum(pdf * np.diff(bins)))

        .. note::

        Until numpy release 1.5, the underlying numpy
        histogram function was incorrect with *normed*=*True*
        if bin sizes were unequal.  MPL inherited that
        error.  It is now corrected within MPL when using
        earlier numpy versions

        *weights*:
        An array of weights, of the same shape as *x*.  Each value in
        *x* only contributes its associated weight towards the bin
        count (instead of 1).  If *normed* is True, the weights are
        normalized, so that the integral of the density over the range
        remains 1.

        *cumulative*:
        If *True*, then a histogram is computed where each bin
        gives the counts in that bin plus all bins for smaller values.
        The last bin gives the total number of datapoints.  If *normed*
        is also *True* then the histogram is normalized such that the
        last bin equals 1. If *cumulative* evaluates to less than 0
        (e.g. -1), the direction of accumulation is reversed.  In this
        case, if *normed* is also *True*, then the histogram is normalized
        such that the first bin equals 1.

        *histtype*: [ 'bar' | 'barstacked' | 'step' | 'stepfilled' ]
        The type of histogram to draw.

        - 'bar' is a traditional bar-type histogram.  If multiple data
        are given the bars are aranged side by side.

        - 'barstacked' is a bar-type histogram where multiple
        data are stacked on top of each other.

        - 'step' generates a lineplot that is by default
        unfilled.

        - 'stepfilled' generates a lineplot that is by default
        filled.

        *align*: ['left' | 'mid' | 'right' ]
        Controls how the histogram is plotted.

        - 'left': bars are centered on the left bin edges.

        - 'mid': bars are centered between the bin edges.

        - 'right': bars are centered on the right bin edges.

        *orientation*: [ 'horizontal' | 'vertical' ]
        If 'horizontal', :func:`~matplotlib.pyplot.barh` will be
        used for bar-type histograms and the *bottom* kwarg will be
        the left edges.

        *rwidth*:
        The relative width of the bars as a fraction of the bin
        width.  If *None*, automatically compute the width. Ignored
        if *histtype* = 'step' or 'stepfilled'.

        *log*:
        If *True*, the histogram axis will be set to a log scale.
        If *log* is *True* and *x* is a 1D array, empty bins will
        be filtered out and only the non-empty (*n*, *bins*,
        *patches*) will be returned.

        *color*:
        Color spec or sequence of color specs, one per
        dataset.  Default (*None*) uses the standard line
        color sequence.

        *label*:
        String, or sequence of strings to match multiple
        datasets.  Bar charts yield multiple patches per
        dataset, but only the first gets the label, so
        that the legend command will work as expected::

        ax.hist(10+2*np.random.randn(1000), label='men')
        ax.hist(12+3*np.random.randn(1000), label='women', alpha=0.5)
        ax.legend()


        kwargs are used to update the properties of the
        :class:`~matplotlib.patches.Patch` instances returned by *hist*:

        agg_filter: unknown
        alpha: float or None         
        animated: [True | False]         
        antialiased or aa: [True | False]  or None for default         
        axes: an :class:`~matplotlib.axes.Axes` instance         
        clip_box: a :class:`matplotlib.transforms.Bbox` instance         
        clip_on: [True | False]         
        clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]         
        color: matplotlib color spec
        contains: a callable function         
        edgecolor or ec: mpl color spec, or None for default, or 'none' for no color         
        facecolor or fc: mpl color spec, or None for default, or 'none' for no color         
        figure: a :class:`matplotlib.figure.Figure` instance         
        fill: [True | False]         
        gid: an id string         
        hatch: [ '/' | '\\' | '|' | '-' | '+' | 'x' | 'o' | 'O' | '.' | '*' ]         
        label: any string         
        linestyle or ls: ['solid' | 'dashed' | 'dashdot' | 'dotted']         
        linewidth or lw: float or None for default         
        lod: [True | False]         
        path_effects: unknown
        picker: [None|float|boolean|callable]         
        rasterized: [True | False | None]         
        snap: unknown
        transform: :class:`~matplotlib.transforms.Transform` instance         
        url: a url string         
        visible: [True | False]         
        zorder: any number 
        """
        if axis == None:
            axis = plt.gca()
        axis.hist(self.__image.ravel(), **kargs)

    def save(self, outputfile, **kargs):
        """
        - Save the image in some common image formats. It uses the pyplot.save 
        method.  
        outputfile is a string containing a path to a filename, 
        of a Python file-like object. If *format* is *None* and
        *fname* is a string, the output format is deduced from
        the extension of the filename.

        Keyword arguments:
        *vmin*/*vmax*: [ None | scalar ]
        *vmin* and *vmax* set the color scaling for the image by fixing the
        values that map to the colormap color limits. If either *vmin* or *vmax*
        is None, that limit is determined from the *arr* min/max value.
        *cmap*:
        cmap is a colors.Colormap instance, eg cm.jet.
        If None, default to the rc image.cmap value.
        *format*:
        One of the file extensions supported by the active
        backend.  Most backends support png, pdf, ps, eps and svg.
        *origin*
        [ 'upper' | 'lower' ] Indicates where the [0,0] index of
        the array is in the upper left or lower left corner of
        the axes. Defaults to the rc image.origin value.
        *dpi*
        The DPI to store in the metadata of the file.  This does not affect the
        resolution of the output image.
        """
        plt.imsave(outputfile, self.__image, **kargs)
