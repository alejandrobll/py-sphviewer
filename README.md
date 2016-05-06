# py-sphviewer [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.21703.svg)](http://dx.doi.org/10.5281/zenodo.21703)



Py-SPHViewer is a publicly available Python package to visualize and explore N-body + Hidrodynamical simulations. The Smoothed Particle Hydrodynamics interpolation scheme is used to interpolate the underlaying density field (or any other property) traced by a finite set of particles, thus producing smooth images of a particle distribution that can be easily interpreted. It does also provide the natural way for visualizing in a self-consistent way gas dynamical simulations in which the same technique is used to compute the particle interactions.

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

<a href="http://www.youtube.com/watch?v=4ZIgVbNlDU4
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

#Changelogs
- version 1.0.4
   * QuickView tool for making quick visualization in just one line (see this [post](https://alejandrobll.wordpress.com/2016/05/04/using-quickview-from-py-sphviewer-1-0-4/) for instructions).
   * I added a new examples directory, which essentially contains a hdf5 file with the coordinates of the particles belonging to a dark matter halo extracted from a cosmological simulation. This is useful to make quick tests. 
   * Minor bugs fixed.


# Code Licence

Py-SPHViewer is under GNU GPL v3 licence, and was started by Alejandro Benitez-Llambay. It is under development and changes everytime. It's your responsibility to be aware of the changes and to check the code for understanding what it does.

# Basic Tutorial

If you are interested in using Py-SPHVIewer, you should take a look at the very little [tutorial](http://nbviewer.ipython.org/urls/raw.githubusercontent.com/alejandrobll/py-sphviewer/master/wiki/tutorial_sphviewer.ipynb). This tutorial assumes that you are familiar with Python. If not, please go to learn Python and come back. 

# Citing the code

If Py-SPHViewer has been significant in your project and leads to a publication, please acknowledge by citing the project. You can check the DOI at Zenodo: [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.21703.svg)](http://dx.doi.org/10.5281/zenodo.21703), or just use the following BibTex:

    @misc{alejandro_benitez_llambay_2015_21703,
    author       = {Alejandro Benitez-Llambay},
    title        = {py-sphviewer: Py-SPHViewer v1.0.0},
    month        = jul,
    year         = 2015,
    doi          = {10.5281/zenodo.21703},
    url          = {http://dx.doi.org/10.5281/zenodo.21703}
    }

or in plain text:

Alejandro Benitez-Llambay. (2015). py-sphviewer: Py-SPHViewer v1.0.0. Zenodo. 10.5281/zenodo.21703

