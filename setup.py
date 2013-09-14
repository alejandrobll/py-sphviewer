from distutils.core import setup

setup(name='Py-SPHViewer',
      version='0.86',
      description='Framework for rendering images using the SPH scheme.',
      author='Alejandro Benitez Llambay',
      author_email='alejandrobll@oac.uncor.edu',
      url='https://code.google.com/p/py-sphviewer/',
      packages=['sphviewer'],
      package_data={'sphviewer': ['*.c','*.txt']},
      license='GNU GPL v3',
      classifiers=[
            "Intended Audience :: Science/Research",
            "License :: GNU GPL v3",
            "Topic :: Scientific/Engineering :: Astronomy",
            "Topic :: Scientific/Engineering :: Visualization",
            "Topic :: Scientific/Engineering :: Rendering"
            ],
      keywords="smoothed particle hydrodynamics render particles nbody galaxy formation dark matter sph cosmology movies",      
     )
