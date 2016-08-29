# py-sphviewer [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.21703.svg)](http://dx.doi.org/10.5281/zenodo.21703)



Py-SPHViewer is a publicly available Python package to visualize and explore N-body + Hidrodynamical simulations. It interpolates an underlaying density field (or any other property) traced by a finite set of particles using the Smoothed Particle Hydrodynamics interpolation scheme, and produces smooth images that can be easily interpreted. It does also provide the natural way for visualizing (in a self-consistent fashion) gas dynamical simulations, which uses the same technique to compute the interactions between particles.

All expensive calculations are permormed by C compiled code, but it can be used interactively from Python, [Ipython](http://ipython.org/) or [Ipython notebook](http://ipython.org/). In addition, Py-SPHViewer allows to allows to explore the simulated volumes using different projections. 
 
Latest versions are normally available in the Python Package Index (or [Pypi](https://pypi.python.org/pypi?:action=display&name=py-sphviewer&version=0.166) for short), and you can get those with:

    pip install py-sphviewer 

or:

    easy_install py-sphviewer 

Note that if you don't have *root* permissions, the --user flag is usually needed.

In case you might want to clone and install the package from this repository, you can simply do:

    git clone https://github.com/alejandrobll/py-sphviewer.git
    cd py-sphviewer
    python setup.py install

Perhaps a good example of the current power of Py-SPHViewer is the logo of the project, 

![Image](https://raw.githubusercontent.com/alejandrobll/py-sphviewer/master/wiki/pysph-logo_small.png)

which shows the dark matter distribution colored according to the velocity dispersion of a small halo taken from a cosmological simulation.

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
   * New QuickView tool for making quick visualizations using just one single line of code (see this [post](https://alejandrobll.wordpress.com/2016/05/04/using-quickview-from-py-sphviewer-1-0-4/) for instructions).
   * I added a new directory called "examples", which essentially contains hdf5 files with simulation outputs. Those are useful to perform quick tests. 
   * Minor bugs fixed.


# Code Licence

Py-SPHViewer is under GNU GPL v3 licence, and was started by Alejandro Benitez-Llambay. It is under development and changes everytime. It's your responsibility to be aware of the changes and check regularly the code to understand what it does.

# Basic Tutorial

If you are interested in using Py-SPHVIewer, you should take a look at this extremely simple [tutorial](http://nbviewer.ipython.org/urls/raw.githubusercontent.com/alejandrobll/py-sphviewer/master/wiki/tutorial_sphviewer.ipynb). This tutorial assumes that you are familiar with Python. If not, please go to learn Python and come back. 
I also try to write different posts highlighting some of the capabilities of Py-SPHViewer. Here I list some of them:

* [Using QuickView to simplify the use of Py-SPHViewer] (https://alejandrobll.wordpress.com/2016/05/04/using-quickview-from-py-sphviewer-1-0-4/)
* [Streamlines with Py-SPHViewer] (https://alejandrobll.wordpress.com/2016/08/29/streamlines-with-py-sphviewer/)

# Citing the code

I encourage people using Py-SPHViewer to cite the project by writing in the acknowledgments:

* This work has benefited from the use of Py-SPHViewer (Benitez-Llambay 2015)

If this package has helped you to gain insight on your simulations, or if it has been significant in your project and leads to a publication, please acknowledge by citing the project. In addition, if your paper contains Figures that have been made with this package, please cite the project. I have created a DOI for this purpose:

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.21703.svg)](http://dx.doi.org/10.5281/zenodo.21703).

You may also want to use the the following BibTex:

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

