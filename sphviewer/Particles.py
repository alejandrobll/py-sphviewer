import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Manager
from scipy.spatial import cKDTree

class Particles():
    def __init__(self, xpos,
                 ypos,
                 zpos,
                 mass = None,
                 hsml = None,
                 rho  = None,
                 prop1 = None,
                 prop2 = None,
                 nb = 32,
                 verbose = False):
        
        self._name = 'PARTICLES'
        self.__pos  = np.array([xpos,ypos,zpos],dtype=np.float32)
        self.__mass = np.array(mass,dtype=np.float32)
        self.__nb   = nb
        self.__verbose = verbose
        self.__prop1 = np.array(prop1,dtype=np.float32)
        self.__prop2 = np.array(prop2,dtype=np.float32)

        if(hsml == None or rho == None):
            self.__hsml = self.__det_hsml(self.__pos,self.__nb)
            self.__rho  = nb/(4./3.*np.pi*self.__hsml**3)
        else:
            self.__hsml = np.array(hsml)
            self.__rho  = np.array(rho)

#Setting methods:
    def set_pos(self,xpos,ypos,zpos):
        self.__pos  = np.array([xpos,ypos,zpos],dtype=np.float32)

    def set_mass(self,mass):
        self.__mass  = np.array(mass,dtype=np.float32)
    
    def set_hsml(self,hsml):
        self.__hsml  = np.array(hsml,dtype=np.float32)

    def set_nb(self,nb):
        self.__nb  = np.array(nb,dtype=np.int32)

    def set_prop1(self,prop1):
        self.__prop1  = np.array(prop1,dtype=np.float32)

    def set_prop2(self,prop2):
        self.__prop2  = np.array(prop2,dtype=np.float32)

#Getting methods
    def get_pos(self):
        """
        x,y,z = get_positions()
        """
        return self.__pos

    def get_mass(self):
        return self.__mass
    
    def get_hsml(self):
        return self.__hsml

    def get_rho(self):
        return self.__rho

    def get_nb(self):
        return self.__nb

    def get_prop1(self):
        return self.__prop1

    def get_prop2(self):
        return self.__prop2

    def plot(self,plane,axis=None,**kargs):
        if(axis == None):
            axis = plt.gca()
        if(plane == 'xy'):
            axis.plot(self.__pos[0,:], self.__pos[1,:], 'k.', **kargs)
        elif(plane == 'xz'):
            axis.plot(self.__pos[0,:], self.__pos[2,:], 'k.', **kargs)
        elif(plane == 'yz'):
            axis.plot(self.__pos[1,:], self.__pos[2,:], 'k.', **kargs)

    def __make_kdtree(self,pos):
        return cKDTree(pos.T)

    def __nbsearch(self, pos, nb, tree, out_hsml, index):
        d, idx = tree.query(pos.T, k=nb)
        out_hsml.put( (index, d[:,nb-1]) )

    def __det_hsml(self, pos, nb):
        """
        Use this function to find out the smoothing length of your particles.
        hsml = det_hsml(pos, nb)
        """
        manager = Manager()
        out_hsml  = manager.Queue()
        size  = multiprocessing.cpu_count()	

        if(self.__verbose): print 'Building a KDTree...'
        tree = self.__make_kdtree(pos)

        index  = np.arange(np.shape(pos)[1])
		#I split the job among the number of available processors
        pos   = np.array_split(pos, size, axis=1)	
        
        procs = []

        #We distribute the tasks among different processes
        if(self.__verbose): print 'Searching the ', nb, 'closer neighbors to each particle...'
        for rank in xrange(size):
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
        for i in xrange(size):
            a, b = out_hsml.get()
            index.append(a)
            hsml.append(b)
    #	    if(a == 0): print b[0]			

            #I have to order the data before return it
        k = np.argsort(index)
        hsml1 = np.array([])
        for i in k:
            hsml1 = np.append(hsml1,hsml[i])
        if(self.__verbose): print 'Done...'
        return hsml1        
