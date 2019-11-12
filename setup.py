#from distutils.core import setup, Extension
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

import numpy as np

module_scene = Extension('sphviewer/extensions/scene', sources = ['sphviewer/extensions/scenemodule.c'],
                         extra_compile_args=['-fopenmp','-w', '-std=c99', '-ffast-math'],
                         extra_link_args=['-lgomp'])

module_render = Extension('sphviewer/extensions/render', sources = ['sphviewer/extensions/rendermodule.c'],
                          extra_compile_args=['-fopenmp','-w', '-std=c99', '-ffast-math'],
                          extra_link_args=['-lgomp'])

module_makehsv = Extension('sphviewer/tools/makehsv', sources = ['sphviewer/tools/makehsvmodule.c'],
                           extra_compile_args=['-fopenmp','-w', '-std=c99', '-ffast-math'],
                           extra_link_args=['-lgomp'])


exec(open('sphviewer/version.py').read())

setup(name='py-sphviewer',
      version=__version__,
      description='Py-SPHViewer is a framework for rendering particles using the smoothed particle hydrodynamics scheme.',
      author='Alejandro Benitez Llambay',
      author_email='alejandrobll@oac.uncor.edu',
      url='https://github.com/alejandrobll/py-sphviewer',
      packages=['sphviewer','sphviewer.extensions', 'sphviewer.tools'],
      include_dirs = [np.get_include()],
      requires = ['pykdtree'],
      install_requires = ['pykdtree'],
      package_data={'sphviewer': ['*.c','*.txt']},
      ext_modules = [module_scene, module_render, module_makehsv],
      license='GNU GPL v3',
      classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Multimedia :: Graphics :: 3D Rendering',
        'Topic :: Multimedia :: Graphics :: Viewers',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Utilities'
            ],
      keywords="smoothed particle hydrodynamics render particles nbody galaxy formation dark matter sph cosmology movies",      
     )
