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
		           hue     = None,
                           sat     = None,
                           nb      = 32, 
                           verbose = False):
#==========================
# particle parameters
		self.pos  = pos
		self.mass = mass
		self.hsml = hsml
		self.hue  = hue
		self.sat  = sat
		self.nb   = nb
#==========================
#==========================
# Camera default parameters
		self.__px      = 0.		# the camera is looking at (px,py,pz)
		self.__py      = 0.		
		self.__pz      = 0.	
		self.__r       = 100.		# the camera is at distance r from (px,py,pz)
		self.__theta   = 0.		# you can rotate the scene using theta and phi
		self.__phi     = 0.		
		self.__zoom    = 1.		# magnification of the camera.
		self.__xsize   = 1000	        # x size of the image
		self.__ysize   = 1000         	# y size of the image
#==========================
		self.__verbose = verbose
		#If smoothing lenghts are not given we compute them.
		if(mass == None):
			if(self.__verbose):
				print "You didn't give me any mass; I will supose unity mass particles"
			self.mass = np.ones(np.shape(pos)[1])
		if(hsml == None): self.hsml = self.__det_hsml(self.pos, self.nb)
		#before to make the render, there is nothing to plot, so:
		self.__is_plot_available = False
		self.__auto_camera()

	def __center_array(self,x):
		xmax = np.max(x)
		xmin = np.min(x)
		xmed = 0.5*(xmax+xmin)
		return (xmax-xmin),xmed

	def __auto_camera(self,distance_factor=0.65):
                """
                Autocentering the camera params
                """
                size_x, self.__px = self.__center_array(self.pos[0,:])
                size_y, self.__py = self.__center_array(self.pos[1,:])
                size_z, self.__pz = self.__center_array(self.pos[2,:])
                self.__r = (distance_factor * 
                          np.sqrt(size_x**2+size_y**2+size_z**2))


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

		if(self.__verbose): print 'Building a KDTree...'
		tree = self.__make_kdtree(pos)

		index  = np.arange(np.shape(pos)[1])
		#I split the job among the number of available processors
		pos   = np.array_split(pos, size, axis=1)	
			
		procs = []

		#We distribute the tasks among different processes
		if(self.__verbose): print 'Searching the ', nb, 'closer neighbors to each particle...'
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
		if(self.__verbose): print 'Done...'
		return hsml1

	def plot(self, logscale=False, square=False, **kargs):
		if(self.__is_plot_available==False):
			print 'There is nothing to plot yet... You should make a render first.'
			return
		dd = self.dens

		if(square):
			dd = self.dens**2
		if(logscale):
			dd = np.log10(dd+1)

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

	def get_camera_params():
		return(self.__px,self.__py,self.__pz,self.__r,self.__theta,
		       self.__phi,self.__zoom,self.__xsize,self.__ysize)

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
		self.__px      = px		
		self.__py      = py		
		self.__pz      = pz	
		self.__r       = r
		self.__theta   = theta		
		self.__phi     = phi
		self.__zoom    = zoom	
		self.__xsize   = xsize
		self.__ysize   = ysize

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
		FOV   = 2.*np.abs(np.arctan(1./(self.__zoom)))

		# we first refer the positions to the camera point of view (px,py,pz) and 
		#then to its physic position (xcam, ycam, zcam). The stright line between
		# (px,py,pz) and (xcam,ycam,zcam) define the line of sight.

		x = self.pos[0,:]-(self.__px)
		y = self.pos[1,:]-(self.__py)
		z = self.pos[2,:]-(self.__pz)

		if(near):
		#we need to take into account the real camera's FOV.
			xmax = self.__zoom * np.tan(FOV/2.)
			xmin = - xmax
			ymax = 0.5*(xmax-xmin)*self.__ysize/self.__xsize	# in order to have symmetric y limits
			ymin = - ymax
			#when near is true we give the extent in angular units
			xfovmax =  FOV/2./ac
			xfovmin =  -FOV/2./ac
			yfovmax = 0.5*(xfovmax-xfovmin)*self.__ysize/self.__xsize
			yfovmin = -yfovmax
			self.extent = np.array([xfovmin,xfovmax,yfovmin,yfovmax])				
		else:
			if(lbox == None):
				xmax = np.max(x)
				xmin = np.min(x)
				ymax = 0.5*(xmax-xmin)*self.__ysize/self.__xsize
				ymin = -ymax
				#when near is False we give the projected coordinates inside lbox 
				self.extent = np.array([xmin+self.__px,
                                                        xmax+self.__px,
                                                        ymin+self.__py,
                                                        ymax+self.__py])				
			else:
				xmax =  lbox/2.
				xmin = -lbox/2.
				ymax = 0.5*(xmax-xmin)*self.__ysize/self.__xsize	# in order to have symmetric y limits
				ymin = - ymax
				#when near is False we give the projected coordinates inside lbox 
				self.extent = np.array([xmin+self.__px,
                                                        xmax+self.__px,
                                                        ymin+self.__py,
                                                        ymax+self.__py])				

		if(self.__theta != 0.):			#we rotate around x axis
			yy_temp       = y*np.cos(self.__theta*ac)+z*np.sin(self.__theta*ac)
			z             = -y*np.sin(self.__theta*ac)+z*np.cos(self.__theta*ac)
			y             = yy_temp
			yy_temp       = 0

		if(self.__phi   != 0.):				#we rotate around y axis
			xx_temp       = x*np.cos(self.__phi*ac)-z*np.sin(self.__phi*ac)
			z             = x*np.sin(self.__phi*ac)+z*np.cos(self.__phi*ac)
			x             = xx_temp
			xx_temp       = 0

		# we now consider only particles in the line of sight inside de FOV of the
                # camera	
		if(near):
			z -= (-1.0*self.__r)	# in order to move the camera far away from the object
			kview = ( np.where( (z > 0.) & (np.abs(x) <= 
		                (np.abs(z)*np.tan(FOV/2.)) ) & (np.abs(y) <= 
		                (np.abs(z)*np.tan(FOV/2.)) ) )[0])

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
		t    = self.hsml[kview]
		if(self.hue != None):
			hue = self.hue[kview]
			sat = self.sat[kview]

		
		lbin = 2*xmax/self.__xsize

		if self.__verbose:
			print '-------------------------------------------------'
			print 'Making the smooth image'
			print 'xmin  =', xmin
			print 'xmax  =', xmax
			print 'ymin  =', ymin
			print 'ymax  =', ymax
			print 'xsize =', self.__xsize
			print 'ysize =', self.__ysize

		if(near):	
			x = ((x*self.__zoom/z-xmin)/(xmax-xmin)*(self.__xsize-1.)).astype(int)
			y = ((y*self.__zoom/z-ymin)/(ymax-ymin)*(self.__ysize-1.)).astype(int)
			t = (t*self.__zoom/z/lbin).astype(int)
		else:
			x = ((x-xmin)/(xmax-xmin)*(self.__xsize-1.)).astype(int)
			y = ((y-ymin)/(ymax-ymin)*(self.__ysize-1.)).astype(int)
			t = (t/lbin).astype(int)
	
		n=int(len(x))

		if(self.hue == None):
			dens = np.zeros([self.__ysize,self.__xsize],dtype=(np.float))
			# C code for making the images
			code       = self.__import_code('c_code.c')
			shared     = (['x','y', 'binx', 'biny',  't', 
                                      'lbin', 'n', 'mass', 'dens'])
		else:
			dens = np.zeros([self.__ysize,self.__xsize,4],dtype=(np.float))
			dens[:,:,3] = 1.
			# C code for making the images
			code       = self.__import_code('c_code_hsv.c')
			shared     = (['x','y', 'binx', 'biny',  't', 
                                      'lbin', 'n', 'mass','hue','sat', 'dens'])


		# interpolation kernel
		extra_code = self.__import_code('extra_code.c')

		binx = self.__xsize
		biny = self.__ysize

		start = time.time()
		weave.inline(code,shared,
                                   support_code=extra_code,
                                   type_converters=converters.blitz,
                                   compiler='gcc', 
                                   extra_compile_args=[' -O3 -fopenmp'],
                                   extra_link_args=['-lgomp'])
		stop = time.time()
		self.dens = dens
		if(self.__verbose): print 'Elapsed time = ', stop-start
		return 
