#from distutils.core import setup, Extension
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

import numpy as np

module_scene = Extension('sphviewer/extensions/scene', sources = ['sphviewer/extensions/scenemodule.c'],
                         extra_compile_args=['-fopenmp'],
                         extra_link_args=['-lgomp'])

module_render = Extension('sphviewer/extensions/render', sources = ['sphviewer/extensions/rendermodule.c'],
                          extra_compile_args=['-fopenmp'],
                          extra_link_args=['-lgomp'])

setup(name='py-sphviewer',
      version='0.166',
      description='Py-SPHViewer is a framework for rendering particles using the smoothed particle hydrodynamics scheme.',
      author='Alejandro Benitez Llambay',
      author_email='alejandrobll@oac.uncor.edu',
      url='https://code.google.com/p/py-sphviewer/',
      packages=['sphviewer','sphviewer.extensions'],
      include_dirs = [np.get_include()],
      requires = ['pykdtree'],
      install_requires = ['pykdtree'],
      package_data={'sphviewer': ['*.c','*.txt']},
      ext_modules = [module_scene, module_render],
      license='GNU GPL v3',
      classifiers=[
        'Programming Language :: Python :: 2.7',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Multimedia :: Graphics :: 3D Rendering',
        'Topic :: Multimedia :: Graphics :: Viewers',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Utilities'
            ],
      keywords="smoothed particle hydrodynamics render particles nbody galaxy formation dark matter sph cosmology movies",      
     )
