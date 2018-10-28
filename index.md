---
title: HomePage
---

Py-SPHViewer is a publicly available Python package to visualise and explore N-body + Hydrodynamics simulations using the Smoothed Particle Hydrodynamics (SPH) scheme. The code estimates the underlying density field (or any other property) traced by a finite number of particles, and produces not only beautiful, but also scientifically useful images. In addition, Py-SPHViewer enables the user to explore simulated volumes using different projections.

Although the package was conceived as a rendering tool for cosmological SPH simulations, it can be used in a number of applications.

Intensive calculations are all performed in parallel C code. However, the package can be used interactively in a Python shell, [Ipython](http://ipython.org/) or [Ipython notebook](http://ipython.org/).


# Installation

The latest stable version of Py-SPHViewer is usually available in the Python Package Index (or [Pypi](https://pypi.python.org/pypi?:action=display&name=py-sphviewer&version=0.166) for short). This is the easiest method to get Py-SPHViewer running in your system and we encourage users to follow it:

    pip install py-sphviewer --user

The development version is available in GitHub. The following lines will clone, compile and install this version:

    git clone https://github.com/alejandrobll/py-sphviewer.git
    cd py-sphviewer
    python setup.py install


# Changelogs

Py-SPHViewer is in active development. We often add new features and fix bugs. We list below the update notes:

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

# Code Licence and citations

 Py-SPHViewer is under GNU GPL v3 licence, and was started by Alejandro Benitez-Llambay. This program is distributed in the hope that it will be useful, but without any warranty.

 Individuals or organizations that use Py-SPHViewer are encourage to cite the code.

 **If Py-SPHViewer has been significant for a research project that leads to a publication, please acknowledge by citing the project and using the following DOI as reference**:

 Alejandro Benitez-Llambay. (2015). py-sphviewer: Py-SPHViewer v1.0.0. Zenodo. 10.5281/zenodo.21703

 You may also use the the following BibTex:

     @misc{alejandro_benitez_llambay_2015_21703,
     author       = {Alejandro Benitez-Llambay},
     title        = {py-sphviewer: Py-SPHViewer v1.0.0},
     month        = jul,
     year         = 2015,
     doi          = {10.5281/zenodo.21703},
     url          = {http://dx.doi.org/10.5281/zenodo.21703}
     }

Some of the scientific papers that used Py-SPHViewer are listed [here](/content/bibliography.html)

# Contributing

Users are encouraged to contribute with ideas, codes or by reporting (and fixing) bugs. Issues and bugs should be reported by creating issues in the main repository. If you would like to contribute with coding, for example, by adding a new feature, tool, or fixing a bug, you are welcome to do so by submitting a pull request.

If you plan to work on a major improvement or a new feature that requires a significant effort from your side, please contact me at alejandro.b.llambay@durham.ac.uk first, so that we can sort out the technical aspects prior starting the development.



# Getting started

To get started with Py-SPHViewer, please go and check the available examples in our Tutorials section. Perhaps the simpler examples that demonstrates the power of Py-SPHViewer viewer is the following:

Download [this hdf5 file](https://github.com/alejandrobll/py-sphviewer/raw/master/examples/darkmatter_box.h5py) and run:

```python
import h5py
from sphviewer.tools import QuickView

with h5py.File('darkmatter_box.h5py','r') as f:
    pdrk = f['PartType1/Coordinates'].value

QuickView(pdrk, r='infinity')
```

which produces the following image:

<p align="center">
   <img src="assets/img/first_image.png" alt="First image with QuickView">
</p>

The result is very impressive given the low-resolution of the simulated volume (only 32768 dark matter particles).

High-resolution cosmological simulations might result in extremely amusing results, such as the one shown in the next [video](https://www.youtube.com/watch?annotation_id=annotation_692472089&feature=iv&src_vid=vqGYURAgYUY&v=4ZIgVbNlDU4):

<p align="center">
   <a href="https://www.youtube.com/watch?annotation_id=annotation_692472089&feature=iv&src_vid=vqGYURAgYUY&v=4ZIgVbNlDU4" target="_blank"><img src="assets/img/video_stars.png" alt="First image with QuickView"> </a>
</p>
