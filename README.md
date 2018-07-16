# py-sphviewer [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.21703.svg)](http://dx.doi.org/10.5281/zenodo.21703)

Py-SPHViewer is a publicly available Python package to visualize and explore N-body + Hydrodynamics simulations. The code interpolates the underlying density field (or any other property) traced by a set of particles, using the Smoothed Particle Hydrodynamics (SPH) interpolation scheme, thus producing not only beautiful, but also useful scientific images. In addition, Py-SPHViewer enables the user to explore simulated volumes using different projections. Finally, Py-SPHViewer provides the natural way to visualize (in a self-consistent fashion) gas dynamical simulations, which use the same technique to compute the interactions between particles.


All expensive calculations are permormed by compiled C code, but the package can be used interactively from a Python shell, [Ipython](http://ipython.org/) or [Ipython notebook](http://ipython.org/). 
 
Latest versions are normally available in the Python Package Index (or [Pypi](https://pypi.python.org/pypi?:action=display&name=py-sphviewer&version=0.166) for short), and you can get those with:

    pip install py-sphviewer 

or:

    easy_install py-sphviewer 

Note that if you don't have *root* permissions, the --user flag is usually needed.

You may want to clone and install the package from this repository. In such a case, simply do:

    git clone https://github.com/alejandrobll/py-sphviewer.git
    cd py-sphviewer
    python setup.py install

Perhaps a good example of the current power of Py-SPHViewer is the logo of the project, 

![Image](https://raw.githubusercontent.com/alejandrobll/py-sphviewer/master/wiki/pysph-logo_small.png)

which shows the dark matter distribution colored according to the velocity dispersion of a small halo taken from a cosmological simulation, or this mock image of a galaxy:

<img src="https://alejandrobll.files.wordpress.com/2015/07/stars-05.jpeg" width="300" height="300">

More examples of the power of Py-SPHViewer are the following movies:

<a href="http://www.youtube.com/watch?v=4ZIgVbNlDU4
" target="_blank"><img src="http://img.youtube.com/vi/vqGYURAgYUY/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

or, 

<a href="https://www.youtube.com/watch?v=2kOMkjETYdU
" target="_blank"><img src="http://img.youtube.com/vi/2kOMkjETYdU/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

or,

<a href="http://www.youtube.com/watch?feature=player_embedded&v=O6Adwk41J58
" target="_blank"><img src="http://img.youtube.com/vi/O6Adwk41J58/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

and, 

<a href="http://www.youtube.com/watch?feature=player_embedded&v=XOcCguGU0cE
" target="_blank"><img src="http://img.youtube.com/vi/XOcCguGU0cE/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

Previous movies shows a 3D rotation around a dark matter halo taken from a cosmological simulation and the "Cosmic Web Stripping" process, which arises from hydrodynamical interaction betweet the gas content of a dwarf galaxy and the large-scale structure of the Universe

# Changelogs

- version 1.1.0
  
  * Py-SPHViewer should work with Python 3. We thank Elliott Sales de Andrade (@QuLogic) for making this possible.
  * Image returned by Render.get_image(), or QuickView.get_image() is normalized by the pixel area. This means that the value of each pixel, when smoothing
    the particle mass, can be regarded as the actual surface density.
    
- version 1.0.5
  * Scipy.weave is now deprecated in Scipy 0.19, so I removed all dependencies to it.
  * Minor bugs fixed.

- version 1.0.4
   * New QuickView tool for making quick visualizations using just one single line of code (see this [post](https://alejandrobll.wordpress.com/2016/05/04/using-quickview-from-py-sphviewer-1-0-4/) for instructions).
   * I added a new directory called "examples", which essentially contains hdf5 files with simulation outputs. Those are useful to perform quick tests. 
   * Minor bugs fixed.


# Code Licence

Py-SPHViewer is under GNU GPL v3 licence, and was started by Alejandro Benitez-Llambay. It is under development and changes everytime. It's your responsibility to be aware of the changes and check regularly the code to understand what it does.

# Basic Tutorial

If you are interested in using Py-SPHVIewer, you should take a look at this extremely simple Tutorial. In addition, I wrote a few posts highlighting some of the capabilities of Py-SPHViewer. Here I list some of them:

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

# Papers that used py-sphviewer:

When people cite the code in their papers, I can keep track of the citation and know what is the real impact of Py-SPHViewer in the scientific community. If your paper used the code and is not listed here, please let me know so that I can add it to the list. Some of the papers that used Py-SPHViewer are the following ones:

* Salcido et al. (2017),  The impact of dark energy on galaxy formation. What does the future of our Universe hold? , arXiv:1710.06861 (https://arxiv.org/abs/1710.06861)
* Benitez-Llambay et al. (2017), The vertical structure of gaseous galaxy discs in cold dark matter halos, arXiv:1707.08046 (https://arxiv.org/abs/1707.08046)
* Leo et al. (2017), The Effect of Thermal Velocities on
Structure Formation in N-body Simulations of Warm Dark Matter, arXiv:1706.07837 (https://arxiv.org/abs/1706.07837)
* Ludlow & Angulo (2016), Einasto Profiles and The Dark Matter Power Spectrum, arXiv:1610.04620 (https://arxiv.org/pdf/1610.04620)
* Benitez-Llambay et al. (2016), The properties of "dark" LCDM halos in the Local Group, arXiv:1609.01301 (https://arxiv.org/abs/1609.01301)
* Algorry et al. (2016), Barred galaxies in the EAGLE cosmological hydrodynamical simulation, arXiv:1609.05909 (https://arxiv.org/pdf/1609.05909)
* Ferrero et al. (2016), Size matters: abundance matching, galaxy sizes, and the Tully-Fisher relation in EAGLE, arXiv:1607.03100 (https://arxiv.org/pdf/1607.03100)
* Benitez-Llambay et al. (2016), Mergers and the outside-in formation of dwarf spheroidals, arXiv:1511.06188 (https://arxiv.org/abs/1511.06188)
* Benitez-Llambay et al. (2015), The imprint of reionization on the star formation histories of dwarf galaxies, arXiv:1405.5540 (https://arxiv.org/abs/1405.5540)
* Algorry et al. (2014), Counterrotating stars in simulated galaxy discs, arXiv:1311.1215 (https://arxiv.org/pdf/1311.1215)
* Benitez-Llambay et al. (2013), Dwarf Galaxies and the Cosmic Web, arXiv:1211.0536 (https://arxiv.org/abs/1211.0536)




