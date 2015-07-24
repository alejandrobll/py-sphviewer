# py-sphviewer

Py-SPHViewer is a publicly available Python package to visualize and explore N-body + Hidrodynamical simulations. The Smoothed Particle Hydrodynamics interpolation scheme is used to interpolate the underlaying density field (or any other property) traced by a finite set of particles, t hus producing smooth images of a particle distribution that can be easily interpreted. It does also provide the natural way for visualizing in a self-consistent way gas dynamical simulations in which the same technique is used to compute the particle interactions.

Part of the code is written in C, but it can be used interactively from Python, using tools such as [Ipython](http://ipython.org/) or [Ipython notebook](http://ipython.org/). It does not only smooth the properties of particles over a ceratin influence domain, but also allows to explore the simulated volumes using different projections. 
 
I usually try to keep the latest (stable or not) version available in the Python Package Index (or [Pypi](https://pypi.python.org/pypi?:action=display&name=py-sphviewer&version=0.166) for short), so that you can get it just doing:

    pip install py-sphviewer 

or:

    easy_install py-sphviewer 

Note that as far as you have not *root* permissions, the --user flag would be requiered.

In case you may want to clone and install by yourself the package from this repository, you can simply do:

    git clone https://github.com/alejandrobll/py-sphviewer.git
    cd py-sphviewer
    python setup.py install

Perhaps a good example of the current power of Py-SPHViewer is the logo of the project, 

![Image](https://raw.githubusercontent.com/alejandrobll/py-sphviewer/master/wiki/pysph-logo_small.png)

The image above shows the dark matter distribution colored according to the velocity dispersion of a small halo taken from a cosmological simulation. It was rendered (of course) with Py-SPHViewer

More examples of the power of Py-SPHViewer are the following movies:

<a href="http://www.youtube.com/watch?v=vqGYURAgYUY
" target="_blank"><img src="http://img.youtube.com/vi/vqGYURAgYUY/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

or, 

<a href="http://www.youtube.com/watch?feature=player_embedded&v=O6Adwk41J58
" target="_blank"><img src="http://img.youtube.com/vi/O6Adwk41J58/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

and, 

<a href="http://www.youtube.com/watch?feature=player_embedded&v=XOcCguGU0cE
" target="_blank"><img src="http://img.youtube.com/vi/XOcCguGU0cE/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

Previous movies show a 3D rotation around a dark matter halo taken from a cosmological simulation and the "Cosmic Web Stripping" process, which arise from hydrodynamical interaction betweet the gas content of dwarf galaxies and the large-scale structure of the Universe, respectively

# Code Licence

Py-SPHViewer is under GNU GPL v3 licence, and was started by Alejandro Benitez-Llambay. It is under development and changes everytime. It's your responsibility to be aware of the changes and to check the code for understanding what it does.

# Basic Tutorial

If you are interested in using Py-SPHVIewer, you should take a look at the very little [tutorial](http://nbviewer.ipython.org/urls/raw.githubusercontent.com/alejandrobll/py-sphviewer/master/wiki/tutorial_sphviewer.ipynb). This tutorial assumes that you are familiar with Python. If not, please go to learn Python and come back. 





