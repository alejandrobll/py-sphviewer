from distutils.core import setup

setup(name='py-sphviewer',
      version='0.88',
      description='Py-SPHViewer is a framework for rendering particles using the smoothed particle hydrodynamics scheme.',
      author='Alejandro Benitez Llambay',
      author_email='alejandrobll@oac.uncor.edu',
      url='https://code.google.com/p/py-sphviewer/',
      packages=['sphviewer'],
      package_data={'sphviewer': ['*.c','*.txt']},
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
