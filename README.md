# Py-SPHViewer [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.21703.svg)](http://dx.doi.org/10.5281/zenodo.21703)


**Py-SPHViewer** is a parallel Python package designed for visualizing and exploring N-body and hydrodynamics simulations using the Smoothed Particle Hydrodynamics (SPH) scheme. The package estimates an underlying scalar field (such as the density field) traced by a finite number of particles, producing not only visually appealing images but also scientifically valuable insights. 

Additionally, **Py-SPHViewer** allows users to explore simulated volumes through various projections. 

Intensive calculations are executed in parallel using C code, which requires OpenMP. As with any Python package, it can be used interactively in a Python shell, [IPython](http://ipython.org/), or [Jupyter Notebook](https://jupyter.org/).

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

# Getting started

To get started with Py-SPHViewer please visit the official website:

<a href="https://alejandrobll.github.io/content/py-sphviewer" target="_blank">**alejandrobll.github.io/content/py-sphviewer**</a>


# Licensing and Citation Information

**Py-SPHViewer** is licensed under the GNU GPL v3 and was initiated by Alejandro Benitez-Llambay. This program is distributed in the hope that it will be useful, but without any warranty.

Individuals or organizations that use **Py-SPHViewer** are encouraged to cite the code.

**If Py-SPHViewer has significantly contributed to a research project leading to a publication, please acknowledge it by citing the project and using the following DOI as a reference**:

Alejandro Benitez-Llambay. (2015). *py-sphviewer: Py-SPHViewer v1.0.0*. Zenodo. [10.5281/zenodo.21703](https://doi.org/10.5281/zenodo.21703)

You may also use the following BibTeX entry:

```bibtex
@misc{alejandro_benitez_llambay_2015_21703,
  author       = {Alejandro Benitez-Llambay},
  title        = {py-sphviewer: Py-SPHViewer v1.0.0},
  month        = jul,
  year         = 2015,
  doi          = {10.5281/zenodo.21703},
  url          = {http://dx.doi.org/10.5281/zenodo.21703}
}
```
