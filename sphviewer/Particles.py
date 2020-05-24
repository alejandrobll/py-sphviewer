#This file is part of Py-SPHViewer

#<Py-SPHVIewer is a framework for rendering particles in Python
#using the SPH interpolation scheme.>
#Copyright (C) <2013>  <Alejandro Benitez Llambay>

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import, division, print_function

import numpy as np
import multiprocessing
from multiprocessing import Manager
from pykdtree.kdtree import KDTree # pykdtree by Esben S. Nielsen 


class Particles(object):
    def __init__(self, pos,
                 mass = None,
                 hsml = None,
                 nb = 32,
                 verbose = False,
                 sort = False):
        """
        The Particles class is a container that stores the essential SPH particle data. Once initialised, this class can be passed
        to the Scene class along with the Camera class to prepare and initialise the scene.  
        
        To instantiate the Particles class, at least one argument with the position of the SPH particles must be provided. In addition, an array containing the mass of the SPH particles, and other array containing smoothing lengths, can be passed. 

        Positions (pos) must be an array of shape [n,3], with
        n being the number of particles. From now on, the initial system 
        of coordinates becomes defined by the particles as follows:

                pos[:,0] = x, pos[:,1] = y and pos[:,2] = z.
        
        If the mass and the smoothing length arrays are not given, it will be  assumed that the masses are all constant with value, mass = 1. The smoothing lengths will be calculated from the distance to the "nb" closer neighbour to each particle. The number of neighbours is set to nb=32 by default. You may want to change this value when instantiating the class for the first time.

        Use *verbose=True* to obtain useful information from the class, mainly for debuggin purposes.

        The *sort* parameter determines whether the SPH particles should be stored in a sorted fashion according to their smoothing lengths. Defining "sort=True" may improve the performance of the rendering of the Scene in some cases, so it is worth considering using it, or at least testing it.

        Once the Particles class has been instantiated, it is not necessary to instantiate it again to pass new updated attributes. The Particles class owns methods for "setting" and "getting" the stored attributes:

        The "setting" methods are:

        - :method:`set_pos(pos)`
        - :method:`set_mass(mass)`
        - :method:`set_hsml(hsml)`
        - :method:`set_nb(nb)`

        The "getting" methods are:
        
        - :method:`get_pos()`
        - :method:`get_mass()`
        - :method:`get_hsml()`
        - :method:`get_nb()`
        """

        self._name = 'PARTICLES'
        self.__verbose = verbose
        npart = np.size(mass)

        if(hsml is None):
            hsml = self.__det_hsml(pos,nb)

        if(sort):
            ksort = np.argsort(hsml)
            self._pos  = pos[ksort,:]
            self._mass = mass[ksort]
            self._hsml = hsml[ksort]
        else:
            self._pos  = pos
            self._mass = mass
            self._hsml = hsml

#Setting methods:
    def set_pos(self,pos):
        """
        Use this method to overwrite the stored positions array. This is useful to update the position of the SPH particles if this is the only attribute that has changed.
        """
        self._pos  = pos

    def set_mass(self,mass):
        """
        Use this method to overwrite the stored masses array. This is useful to update the masses of the SPH particles if this is the only attribute that has changed.
        """
        self._mass  = mass
    
    def set_hsml(self,hsml):
        """
        Use this method to overwrite the stored smoothing lengths array. This is useful to update the smoothing lengths of the SPH particles if this is the only attribute that has changed.
        """
        self._hsml  = hsml

    def set_nb(self,nb):
        """
        Use this method to overwrite the value of nb, i.e., the number of neighbours considered to estimate the SPH smoothing lengths. Using this method trigers the calculation of smoothing lenghts, which may be an expensive calculation.
        """
        self.__nb  = nb
        self.__hsml = self.__det_hsml(self.__pos,self.__nb)


#Getting methods
    def get_pos(self):
        """
        Use this method to get the stored positions of the SPH particles.
        """
        return self._pos

    def get_mass(self):
        """
        Use this method to get the stored masses of the SPH particles.
        """
        return self._mass
    
    def get_hsml(self):
        """
        Use this method to get the stored smoothing lengths of the SPH particles.
        """
        return self._hsml

    def get_nb(self):
        """
        Use this method to get the value of nb, i.e., the number of neighbours considered to estimate the SPH smoothing lengths.
        """
        return self.__nb

    def __make_kdtree(self,pos):
        return KDTree(pos)
    
    def __nbsearch(self, pos, nb, tree):
        d, idx = tree.query(pos, k=nb)
        hsml = d[:,nb-1]
        return hsml
    
    def __det_hsml(self, pos, nb):
        tree = self.__make_kdtree(pos)
        hsml = self.__nbsearch(pos, nb, tree)
        return hsml

    def __make_kdtree_old(self,pos):
        from scipy.spatial import cKDTree
        return cKDTree(pos)

    def __nbsearch_old(self, pos, nb, tree, out_hsml, index):
        d, idx = tree.query(pos, k=nb)
        out_hsml.put( (index, d[:,nb-1]) )

    def __det_hsml_old(self, pos, nb):
        """
        Use this function to find the smoothing lengths of the particles.
        hsml = det_hsml(pos, nb)
        """
        manager = Manager()
        out_hsml  = manager.Queue()
        size  = multiprocessing.cpu_count()	

        if self.__verbose:
            print('Building a KDTree...')
        tree = self.__make_kdtree(pos)

        index  = np.arange(np.shape(pos)[1])
		#I split the job among the number of available processors
        pos   = np.array_split(pos, size, axis=1)	
        
        procs = []

        #We distribute the tasks among different processes
        if self.__verbose:
                print('Searching the ', nb,
                      'closer neighbors to each particle...')
        for rank in range(size):
            task = multiprocessing.Process(target=self.__nbsearch, 
                                           args=(pos[rank], nb, tree, 
                                                 out_hsml,rank))
            procs.append(task) 
            task.start()
            
            #Wait until all processes finish
        for p in procs:
            p.join()

            index = []
            hsml  = []
        for i in range(size):
            a, b = out_hsml.get()
            index.append(a)
            hsml.append(b)
    #	    if a == 0: print(b[0])

            #I have to order the data before return it
        k = np.argsort(index)
        hsml1 = np.array([])
        for i in k:
            hsml1 = np.append(hsml1,hsml[i])
        if self.__verbose:
            print('Done...')
        return hsml1        
