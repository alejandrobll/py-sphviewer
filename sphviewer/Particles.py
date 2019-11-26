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
import matplotlib.pyplot as plt
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
        Particles class stores particle's information that will be used 
        later for rendering the image. 
        
        Input arguments are: particles position, mass and optionally smoothing lengths. 
        
        Positions of the particles are given by the array *pos* with shape [3,n], where
        n is the number of particles, and pos[0,:] = x, pos[1,:] = y and pos[2,:] = z. 
        
        If masses and smoothing lengths are not given, Particles class assumes the same mass = 1 for every particle, 
        and the corresponding smoothing length of each particle is computed as the distance to its "nb" closer neighbor. 
        By default nb=32. 

        Note that once you have created an instance of Particle, it is not necessary
        to instantiate it again in case you may want to change any property. 
        Particles class has its own method for setting and/or getting
        the properties of the particles already stored:

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

        Finally, Particles class has its own plotting method:

        - :method:`plot('plane', axis=None, **kwargs)`

        in which 'plane' is one of the available projections of the input data:
        |'xy'|'xz'|'yz'|, and axis makes a reference to an existing axis. 
        If axis is None (default), the plot is made on the current active axis. 
        
        Please read the matplotlib.pyplot.plot documentation for the accepted
        The kwargs are :class:`~matplotlib.lines.Line2D` properties:
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
        Use this method to overwrite the stored array of particles.
        """
        self._pos  = pos

    def set_mass(self,mass):
        """
        Use this method to overwrite the stored array of masses.
        """
        self._mass  = mass
    
    def set_hsml(self,hsml):
        """
        Use this method to overwrite the stored array of smoothing lengths.
        """
        self._hsml  = hsml

    def set_nb(self,nb):
        """
        Use this method to overwrite the defined number of neighbors
        to be used for computing the smoothing lengths.
        """
        self.__nb  = nb

#Getting methods
    def get_pos(self):
        """
        Use this method to get the stored array of particles.
        - Output: [3,n] numpy array with x = pos[0,:], y = pos[1,:], z = pos[2,:]
        with n the number of particles.
        """
        return self._pos

    def get_mass(self):
        """
        Use this method to get the stored array of masses.
        """
        return self._mass
    
    def get_hsml(self):
        """
        Use this method to get the stored array of smoothing lengths.
        """
        return self._hsml

    def get_nb(self):
        """
        Use this method to get the number of neighbors that were used to 
        compute the smoothing lengths.
        """
        return self.__nb

    def plot(self,plane,axis=None,**kargs):
        """
        Use this method to plot the set of particles stored by the Particles class.
        In order to plot the distribution of Particles, a *plane* parameter must be given.
        "plane" is one of the available orthogonal projections of the particles:  
        |'xy'|'xz'|'yz'|. If there is multiple axes defined, the active one can be 
        selected using the axis parameter. If axis paremeter is None (default), the 
        distribution of particles is plotted in the active axis, which is returned by 
        the matplotlib.pyplot.gca() method.
        """
        if(axis == None):
            axis = plt.gca()
        if(plane == 'xy'):
            axis.plot(self._pos[:,0], self._pos[:,0], 'k.', **kargs)
        elif(plane == 'xz'):
            axis.plot(self._pos[:,1], self._pos[:,2], 'k.', **kargs)
        elif(plane == 'yz'):
            axis.plot(self._pos[:,2], self._pos[:,2], 'k.', **kargs)

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
