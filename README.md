# py-sphviewer

Py-SPHViewer is a publicly availabel Python package to visualize and explore N-body + Hidrodynamical simulations. The Smoothed Particle Hydrodynamics interpolation scheme is used to interpolate the underlaying density field (or any other property) traced by a finite set of particles, thus producing smooth images of particle distribution that can be easily interpreted. It does also provide the natural way for visualizing in a self-consistent way gas dynamical simulations in which the same technique is used to compute the particle interactions.

Part of the code is written in C, but it can be used interactively from Python, using tools such as [Ipython](http://ipython.org/)
 or [Ipython notebook](http://ipython.org/). It does not only smooth the properties of particles over a ceratin influence domain, but also allows to explore the simulated volumes using different projections. 
 
I try to keep the latest stable (or not) version available in the Python Package Index (or [Pypi](https://pypi.python.org/pypi?:action=display&name=py-sphviewer&version=0.166) for short), so that you can get it just doing:

    pip install py-sphviewer 

or:
    easy_install py-sphviewer 

Note that as far as you have not *root* permissions, the --user flag would be requiered.

In case you may want to clone and install by yourself the package from this repository, you can simply do:

    git clone https://github.com/alejandrobll/py-sphviewer.git
    cd py-sphviewer
    python setup.py install
    cd ..
    python
    >>> import sphviewer
    

Perhaps a good example of the current power of Py-SPHViewer is the logo of the project, 



![Image](https://raw.githubusercontent.com/alejandrobll/py-sphviewer/master/wiki/pysph-logo_small.png)

The image above shows the dark matter distribution colored according to the velocity dispersion o a small halo taken from a cosmological simulation. It was rendered (of course) with Py-SPHViewer
