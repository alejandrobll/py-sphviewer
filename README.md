# Py-SPHViewer [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.21703.svg)](http://dx.doi.org/10.5281/zenodo.21703)

Py-SPHViewer is a publicly available Python package to visualise and explore N-body + Hydrodynamics simulations using the Smoothed Particle Hydrodynamics (SPH) scheme. The code estimates the underlying density field (or any other property) traced by a finite number of particles, and produces not only beautiful, but also scientifically useful images. In addition, Py-SPHViewer enables the user to explore simulated volumes using different projections.

Although the package was conceived as a rendering tool for cosmological SPH simulations, it can be used in a number of applications.

Intensive calculations are all performed in parallel C code. However, the package can be used interactively in a Python shell, [Ipython](http://ipython.org/) or [Ipython notebook](http://ipython.org/). 

# Installation

The latest stable version of Py-SPHViewer is usually available in the Python Package Index (or Pypi for short). This is the easiest method to get Py-SPHViewer running in your system and we encourage users to follow it:

```
pip install py-sphviewer --user
```
Py-SPHViewer can also be installed using conda:

```
conda install -c alejandrobll py-sphviewer
```

The development version can be cloned, compiled and installed using the lines below:

```
git clone https://github.com/alejandrobll/py-sphviewer.git
cd py-sphviewer
python setup.py install
```

# Changelogs

Py-SPHViewer is in active development. We often add new features and fix bugs. We list below the update notes:

- **version 1.2.1**
  * Bug with convergence for low-resolution images fixed. We thank Josh Borrow for pointing the fix. We note that there is still an ongoing issue that prevents image convergence to better than 3% for intermediate to low resolution (when the physical smoothing of most particles overlaps very few pixels). If you need to reconstruct the field to better than 3%, please perform a convergence test and indentify the resolution above which you feel safe.
  * ``Render.get_image()`` now returns the actual surface density as in version 1.1.0 (this was dropped in v1.2).

- **version 1.2.0**
  * Major change in the way particle data is stored and accessed by Scene. The result is a significant speed up of the code
  * Particles data must be of shape [N,3], in constrast with [3,N] that was used in earlier versions.

- **version 1.1.0**

  * Py-SPHViewer is now compatible with Python 3. We thank Elliott Sales de Andrade (@QuLogic) for making this possible.
  * Image returned by Render.get_image(), or QuickView.get_image() is normalized by the pixel area. This means that the value of each pixel, when smoothing the particle mass, can be regarded as the actual surface density.

- **version 1.0.5**

  * Scipy.weave is deprecated in Scipy 0.19, so we removed all dependencies to it.
  * Minor bugs fixed.

- **version 1.0.4**

   * New QuickView tool for making quick visualisations (see this [post](https://sites.google.com/view/abll/codes/py-sphviewer/using-quickview) for instructions).
   * We added a new directory "examples" that contains hdf5 files with simulation outputs. These are useful to perform quick tests.
   * Minor bugs fixed.
   

# Getting started

To get started with Py-SPHViewer please visit the official website:

[**alejandrobll.github.io/py-sphviewer**](https://alejandrobll.github.io/py-sphviewer)


# Acknowledging the code

Py-SPHViewer is under GNU GPL v3 licence, and was started by Alejandro Benitez-Llambay. This program is distributed in the hope that it will be useful, but without any warranty.

Individuals or organizations that use Py-SPHViewer are encouraged to cite the code.

If Py-SPHViewer has been significant for a research project that leads to a publication, **please acknowledge** by citing the project and using the following DOI as reference:

**Alejandro Benitez-Llambay. (2015). py-sphviewer: Py-SPHViewer v1.0.0. Zenodo. 10.5281/zenodo.21703**

You may also use the the following BibTex:

```
 @misc{alejandro_benitez_llambay_2015_21703,
 author       = {Alejandro Benitez-Llambay},
 title        = {py-sphviewer: Py-SPHViewer v1.0.0},
 month        = jul,
 year         = 2015,
 doi          = {10.5281/zenodo.21703},
 url          = {http://dx.doi.org/10.5281/zenodo.21703}
 }
```

Some of the scientific papers that used Py-SPHViewer are listed [**here**](http://alejandrobll.github.io/py-sphviewer/content/bibliography.html)

# Contributing

Users are encouraged to contribute with ideas, codes or by reporting (and fixing) bugs. 

Issues and bugs should be reported by creating issues in the main repository. If you would like to contribute with code that adds new feature, tool, or that fixes a bug, you are very welcome to do so by submitting a pull request.

If you plan to work on a major improvement, or on a new feature that requires a significant effort from your side, please contact me at alejandro.b.llambay@durham.ac.uk first so that we can sort out the technical aspects prior starting the development.
