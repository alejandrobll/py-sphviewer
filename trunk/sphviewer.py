import numpy as np
import sys
import time
import multiprocessing
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import weave
from scipy.weave import converters
from multiprocessing import Manager


class init(object):
	"""
	Main Class to render the particles. This class must be instantiated by
	specifying at least the position of the particles in an array of
	dimension [3,n], where n in the number of particles. If you do not
	specify the smoothing lenght nither the density, the class will compute
	them by itself. By default, this class compute a variable smoothing lenght
	according to the distance to the nb (nb = 32 by default) closer neighbor. 
	By default, sphviewer re-center the scene (see center_array function). If 
	it does'n work well, you should set ac=False, and re-center the scene 
	manually after, in the camera_params. 

	Please, report any problem o bug to alejandrobll@oac.uncor.edu
	"""
	def __init__(self, pos     = None, 
                           mass    = None, 
                           hsml    = None, 
                           nb      = 32, 
                           verbose = False):
#==========================
# particle parameters
		self.pos  = pos
		self.mass = mass
		self.hsml = hsml
		self.nb   = nb
#==========================
#==========================
# Camera default parameters
		self.px      = 0.		# the camera is looking at (px,py,pz)
		self.py      = 0.		
		self.pz      = 0.	
		self.r       = 100.		# the camera is at distance r from (px,py,pz)
		self.theta   = 0.		# you can rotate the scene using theta and phi
		self.phi     = 0.		
		self.zoom    = 1.		# magnification of the camera.
		self.xsize   = 1000	        # x size of the image
		self.ysize   = 1000         	# y size of the image
#==========================
		self.verbose = verbose
		#If smoothing lenghts are not given we compute them.
		if(mass == None):
			if(self.verbose):
				print "You didn't give me any mass; I will supose unity mass particles"
			self.mass = np.ones(np.shape(pos)[1])
		if(hsml == None): self.hsml = self.__det_hsml(self.pos, self.nb)
		#before to make the render, there is nothing to plot, so:
		self.__is_plot_available = False


	def __import_code(self,filename):
		#naive function to import c code
		string = ''
		fi = open(filename, 'r').readlines()
		for i in fi:
			string += i
		return string


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

		if(self.verbose): print 'Building a KDTree...'
		tree = self.__make_kdtree(pos)

		index  = np.arange(np.shape(pos)[1])
		#I split the job among the number of available processors
		pos   = np.array_split(pos, size, axis=1)	
			
		procs = []

		#We distribute the tasks among different processes
		if(self.verbose): print 'Searching the ', nb, 'closer neighbors to each particle...'
		for rank in xrange(size):
			task = multiprocessing.Process( 
		                target=self.__nbsearch, 
		                args=(pos[rank],
		                      nb, tree, out_hsml, 
		                      rank))
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
#			if(a == 0): print b[0]			

		#I have to order the data before return it
		k = np.argsort(index)
		hsml1 = np.array([])
		for i in k:
			hsml1 = np.append(hsml1,hsml[i])
		if(self.verbose): print 'Done...'
		return hsml1#, nb/(4./3.*np.pi*hsml1**3)

	def plot(self, logscale=False, **kargs):
		if(self.__is_plot_available==False):
			print 'There is nothing to plot yet... You should make a render first.'
			return

		if(logscale):
			dd = np.log10(self.dens+1)
		else:
			dd = self.dens

		plt.imshow(dd, origin='lower', extent=self.extent, **kargs)
		plt.show()

	def get_min(self):
		if(self.__is_plot_available):
			return np.min(self.dens)
		else:
			print 'There is nothing to show...'
			return

	def get_max(self):
		if(self.__is_plot_available):
			return np.max(self.dens)
		else:
			print 'There is nothing to show...'
			return

	def save(self, fname, **kargs):
		plt.imsave(fname,self.dens,**kargs)

	
	def set_camera_params(self, px    = 0.,
                                    py    = 0.,
                                    pz    = 0., 
                                    r     = 100.,
                                    theta = 0.,
                                    phi   = 0.,
                                    zoom  = 1.,
                                    xsize = 1000,
                                    ysize = 1000):
		"""
		Use camera_params to define the (px,py,pz) looking point of the camera,
		distance "r" of the observer, the angles "theta" and "phi" of camera, the 
		"zoom" and resolution of the image.
		"""
		self.px      = px		
		self.py      = py		
		self.pz      = pz	
		self.r       = r
		self.theta   = theta		
		self.phi     = phi
		self.zoom    = zoom	
		self.xsize   = xsize
		self.ysize   = ysize

		if(self.verbose):
			print '\n==============================='
			print '==== Parameters of camera: ======'
			print '(px,py,pz)           = ',     self.px,',',self.py,',',self.pz
			print 'r                    = ',     self.r
			print 'theta                = ',     self.theta
			print 'phi                  = ',     self.phi
			print 'zoom                 = ',     self.zoom
			print 'xsize                = ',     self.xsize
			print 'ysize                = ',     self.ysize
			print '================================'


	def render(self, near = True,
                         lbox = None):
		#define near as False if you want to look at the scene from the infinity
		#lbox defines the physical lenght showed by the camera when near is False

#		if( (near == False) & (lbox == None) ):
#			print 'ERROR: If you selected near=False, you have to give some lbox value...'

		#is there anything to plot?
		self.__is_plot_available = True

		#factor to convert angles
		ac = np.pi/180.
		FOV   = 2.*np.abs(np.arctan(1./(self.zoom)))

		if self.verbose:
			#we write the camera parameters for clarity
			print '\n==============================='
			print '==== Parameters of camera: ===='
			print '(px,py,pz)           = ',     self.px,',',self.py,',',self.pz
			print 'r                    = ',     self.r
			print 'theta                = ',     self.theta
			print 'phi                  = ',     self.phi
			print 'zoom                 = ',     self.zoom
			print 'xsize                = ',     self.xsize
			print 'ysize                = ',     self.ysize
			print '================================'
		# we first refer the positions to the camera point of view (px,py,pz) and 
		#then to its physic position (xcam, ycam, zcam). The stright line between
		# (px,py,pz) and (xcam,ycam,zcam) define the line of sight.

		x = self.pos[0,:]-(self.px)
		y = self.pos[1,:]-(self.py)
		z = self.pos[2,:]-(self.pz)

		if(near):
		#we need to take into account the real camera's FOV.
			xmax = self.zoom * np.tan(FOV/2.)
			xmin = - xmax
			ymax = 0.5*(xmax-xmin)*self.ysize/self.xsize	# in order to have symmetric y limits
			ymin = - ymax
			#when near is true we give the extent in angular units
			xfovmax =  FOV/2./ac
			xfovmin =  -FOV/2./ac
			yfovmax = 0.5*(xfovmax-xfovmin)*self.ysize/self.xsize
			yfovmin = -yfovmax
			self.extent = np.array([xfovmin,xfovmax,yfovmin,yfovmax])				
		else:
			if(lbox == None):
				xmax = np.max(x)
				xmin = np.min(x)
				ymax = 0.5*(xmax-xmin)*self.ysize/self.xsize
				ymin = -ymax
				#when near is False we give the projected coordinates inside lbox 
				self.extent = np.array([xmin+self.px,
                                                        xmax+self.px,
                                                        ymin+self.py,
                                                        ymax+self.py])				
			else:
				xmax =  lbox/2.
				xmin = -lbox/2.
				ymax = 0.5*(xmax-xmin)*self.ysize/self.xsize	# in order to have symmetric y limits
				ymin = - ymax
				#when near is False we give the projected coordinates inside lbox 
				self.extent = np.array([xmin+self.px,
                                                        xmax+self.px,
                                                        ymin+self.py,
                                                        ymax+self.py])				

		if(self.theta != 0.):			#we rotate around x axis
			yy_temp       = y*np.cos(self.theta*ac)+z*np.sin(self.theta*ac)
			z             = -y*np.sin(self.theta*ac)+z*np.cos(self.theta*ac)
			y             = yy_temp
			yy_temp       = 0

		if(self.phi   != 0.):				#we rotate around y axis
			xx_temp       = x*np.cos(self.phi*ac)-z*np.sin(self.phi*ac)
			z             = x*np.sin(self.phi*ac)+z*np.cos(self.phi*ac)
			x             = xx_temp
			xx_temp       = 0

		# we now consider only particles in the line of sight inside de FOV of the
                # camera	
		if(near):
			z -= (-1.0*self.r)	# in order to move the camera far away from the object
			kview = ( np.where( (z > 0.) & (np.abs(x) <= 
		                (np.abs(z)*np.tan(FOV/2.)) ) & (np.abs(y) <= 
		                (np.abs(z)*np.tan(FOV/2.)) ) )[0])

			x = x[kview]
			y = y[kview]
			z = z[kview]
			mass = self.mass[kview]
		else:
			if(lbox == None):
				kview = np.where( (x >= xmin) & (x <= xmax) & 
		                                  (y >= ymin) & (y <= ymax) )[0]
			else:
				kview = np.where( (x >= xmin) & (x <= xmax) & 
		                                  (y >= ymin) & (y <= ymax) &
		                                  ( np.abs(z) <= lbox/2 ) )[0]
				print np.min(np.abs(z)), np.max(np.abs(z))
			x = x[kview]
			y = y[kview]
			z = z[kview]

			mass = self.mass[kview]

		lbin = 2*xmax/self.xsize

		if self.verbose:
			print '-------------------------------------------------'
			print 'Making the smooth image'
			print 'xmin  =', xmin
			print 'xmax  =', xmax
			print 'ymin  =', ymin
			print 'ymax  =', ymax
			print 'xsize =', self.xsize
			print 'ysize =', self.ysize

		if(near):	
			x = ((x*self.zoom/z-xmin)/(xmax-xmin)*(self.xsize-1.)).astype(int)
			y = ((y*self.zoom/z-ymin)/(ymax-ymin)*(self.ysize-1.)).astype(int)
			t = (self.hsml[kview]*self.zoom/z/lbin).astype(int)
		else:
			x = ((x-xmin)/(xmax-xmin)*(self.xsize-1.)).astype(int)
			y = ((y-ymin)/(ymax-ymin)*(self.ysize-1.)).astype(int)
			t = (self.hsml[kview]/lbin).astype(int)
	
		n=int(len(x))

		dens = np.zeros([self.ysize,self.xsize],dtype=(np.float))

		# interpolation kernel
		extra_code = self.__import_code('extra_code.c')
		# C code for making the images
		code       = self.__import_code('c_code.c')

		binx = self.xsize
		biny = self.ysize

		start = time.time()
		weave.inline(code,['x',
                                   'y',
                                   'binx',
                                   'biny', 
                                   't',
                                   'lbin',
                                   'n',
                                   'mass',
                                   'dens'],
                                   support_code=extra_code,
                                   type_converters=converters.blitz,
                                   compiler='gcc', 
                                   extra_compile_args=[' -O3 -fopenmp'],
                                   extra_link_args=['-lgomp'])
		stop = time.time()
		self.dens = dens
		if(self.verbose): print 'Elapsed time = ', stop-start
		return 
