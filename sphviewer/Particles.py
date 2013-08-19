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
                 nb = 32,
                 verbose = False):
        """
        This class allows to load all relevant particles properties.
        If positions "xpos", "ypos" and "zpos" are given, this class
        computes the smoothing lenghts of each particle by using the
        distance to the "nb" neighbor. By default nb=32. 

        Note that once you have created a Particle object, you don't
        need to define it again in case you want to change some property.
        Particles class has its own method for setting and/or getting
        the properties of the particles:

        Setting method are:

        - set_pos()
        - set_mass()
        - set_hsml()
        - set_nb()

        Getting methods are:
        
        - get_pos()
        - get_mass()
        - get_hsml()
        - get_nb()

        Finally, Particles class has its own plotting method:

        - plot('plane', axis=None, *kwargs)

        'plane' is one of the available projections of the input data:
        |'xy'|'xz'|'yz'|.
        axis makes a reference to an existing axis. In case axis is None,
        the plot is made on the current axis. 

        The kwargs are :class:`~matplotlib.lines.Line2D` properties:

        agg_filter: unknown
        alpha: float (0.0 transparent through 1.0 opaque)         
        animated: [True | False]         
        antialiased or aa: [True | False]         
        axes: an :class:`~matplotlib.axes.Axes` instance         
        clip_box: a :class:`matplotlib.transforms.Bbox` instance         
        clip_on: [True | False]         
        clip_path: [ (:class:`~matplotlib.path.Path`,         :class:`~matplotlib.transforms.Transform`) |         :class:`~matplotlib.patches.Patch` | None ]         
        color or c: any matplotlib color         
        contains: a callable function         
        dash_capstyle: ['butt' | 'round' | 'projecting']         
        dash_joinstyle: ['miter' | 'round' | 'bevel']         
        dashes: sequence of on/off ink in points         
        data: 2D array (rows are x, y) or two 1D arrays         
        drawstyle: [ 'default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post' ]         
        figure: a :class:`matplotlib.figure.Figure` instance         
        fillstyle: ['full' | 'left' | 'right' | 'bottom' | 'top']         
        gid: an id string         
        label: any string         
        linestyle or ls: [ ``'-'`` | ``'--'`` | ``'-.'`` | ``':'`` | ``'None'`` | ``' '`` | ``''`` ]         and any drawstyle in combination with a linestyle, e.g. ``'steps--'``.         
        linewidth or lw: float value in points         
        lod: [True | False]         
        marker: [ ``7`` | ``4`` | ``5`` | ``6`` | ``'o'`` | ``'D'`` | ``'h'`` | ``'H'`` | ``'_'`` | ``''`` | ``'None'`` | ``' '`` | ``None`` | ``'8'`` | ``'p'`` | ``','`` | ``'+'`` | ``'.'`` | ``'s'`` | ``'*'`` | ``'d'`` | ``3`` | ``0`` | ``1`` | ``2`` | ``'1'`` | ``'3'`` | ``'4'`` | ``'2'`` | ``'v'`` | ``'<'`` | ``'>'`` | ``'^'`` | ``'|'`` | ``'x'`` | ``'$...$'`` | *tuple* | *Nx2 array* ]
        markeredgecolor or mec: any matplotlib color         
        markeredgewidth or mew: float value in points         
        markerfacecolor or mfc: any matplotlib color         
        markerfacecoloralt or mfcalt: any matplotlib color         
        markersize or ms: float         
        markevery: None | integer | (startind, stride)
        picker: float distance in points or callable pick function         ``fn(artist, event)``         
        pickradius: float distance in points         
        rasterized: [True | False | None]         
        snap: unknown
        solid_capstyle: ['butt' | 'round' |  'projecting']         
        solid_joinstyle: ['miter' | 'round' | 'bevel']         
        transform: a :class:`matplotlib.transforms.Transform` instance         
        url: a url string         
        visible: [True | False]         
        xdata: 1D array         
        ydata: 1D array         
        zorder: any number         
        
        kwargs *scalex* and *scaley*, if defined, are passed on to
        :meth:`~matplotlib.axes.Axes.autoscale_view` to determine
        whether the *x* and *y* axes are autoscaled; the default is
        *True*.
        
        Additional kwargs: hold = [True|False] overrides default hold state
        """

        
        self._name = 'PARTICLES'
        self.__pos  = np.array([xpos,ypos,zpos],dtype=np.float32)
        self.__mass = np.array(mass,dtype=np.float32)
        self.__nb   = nb
        self.__verbose = verbose

        if(hsml == None):
            self.__hsml = self.__det_hsml(self.__pos,self.__nb)
        else:
            self.__hsml = np.array(hsml)

#Setting methods:
    def set_pos(self,xpos,ypos,zpos):
        self.__pos  = np.array([xpos,ypos,zpos],dtype=np.float32)

    def set_mass(self,mass):
        self.__mass  = np.array(mass,dtype=np.float32)
    
    def set_hsml(self,hsml):
        self.__hsml  = np.array(hsml,dtype=np.float32)

    def set_nb(self,nb):
        self.__nb  = np.array(nb,dtype=np.int32)

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

    def get_nb(self):
        return self.__nb

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
