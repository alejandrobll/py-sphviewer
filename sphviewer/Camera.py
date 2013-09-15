import numpy as np
import matplotlib.pylab as plt

class Camera():
    def __init__(self, x = None, y = None, z = None, r = None, 
                 t = None, p = None, zoom  = None, 
                 xsize = None, ysize = None):
        self._name = 'CAMERA'        
        self.__params = {'x':x,'y':y,'z':z,'r':r,
                         't':t,'p':p,'zoom':zoom,
                         'xsize':xsize, 'ysize':ysize}

    def get_params(self):
        return self.__params
    
    def set_params(self, **kargs):
            for key in kargs:
                self.__params[key] = kargs[key]

    def _get_camera(self,plane, **kargs):
        x0 = self.__params['r']*(np.sin(self.__params['p']*np.pi/180.)*
                                 np.cos(self.__params['t']*np.pi/180.))
        z0 = self.__params['r']*(np.cos(self.__params['p']*np.pi/180.)*
                                np.cos(self.__params['t']*np.pi/180.))
        y0 = self.__params['r']*np.sin(self.__params['t']*np.pi/180.)
        
        xcam = self.__params['x']+x0
        ycam = self.__params['y']+y0
        zcam = self.__params['z']+z0
        
        
        if(plane == 'xy'):
            camera = plt.Line2D([xcam],[ycam], c='m', marker='o', markersize=15, **kargs)
            arrow =  plt.Line2D([xcam,self.__params['x']],[ycam,self.__params['y']],lw=5)
        elif(plane == 'xz'):
            camera = plt.Line2D([xcam],[zcam], c='m', marker='o', markersize=15, **kargs)
            arrow =  plt.Line2D([xcam,self.__params['x']],[zcam,self.__params['z']],lw=5)
        elif(plane == 'yz'):
            camera = plt.Line2D([ycam],[zcam], c='m', marker='o', markersize=15, **kargs)
            arrow =  plt.Line2D([ycam,self.__params['y']],[zcam,self.__params['z']],lw=5)
        else:
            print 'Incorrect plane:', plane
            print "Possibles planes are: 'xy';'xz';'yz'"
            return
        return camera, arrow

    def plot(self,plane, axis=None, **kargs):
        if(axis == None):
            axis = plt.gca()
        camera, arrow = self._get_camera(plane)
        axis.add_line(camera, **kargs)
        axis.add_line(arrow, **kargs)

    def set_autocamera(self,Particles, mode='density'):
        try:
            particles_name = Particles._name
        except AttributeError:
            print "You must use a valid class..."
            return
        if (particles_name != 'PARTICLES'):
            print "You must use a valid Particles class..."
            return
        xmax,ymax,zmax = (np.max(Particles.get_pos()[0,:]),
                          np.max(Particles.get_pos()[1,:]),
                          np.max(Particles.get_pos()[2,:]))
        xmin,ymin,zmin = (np.min(Particles.get_pos()[0,:]),
                          np.min(Particles.get_pos()[1,:]),
                          np.min(Particles.get_pos()[2,:]))

        if(mode == 'minmax'):
            xmean = (xmax+xmin)/2.
            ymean = (ymax+ymin)/2.
            zmean = (zmax+zmin)/2.
            
        if(mode == 'density'):
            k = np.argmin(Particles.get_hsml()[:])
            xmean = Particles.get_pos()[0,k]
            ymean = Particles.get_pos()[1,k]
            zmean = Particles.get_pos()[2,k]

        if(mode == 'median'):
            xmean = np.median(Particles.get_pos()[0,:])
            ymean = np.median(Particles.get_pos()[1,:])
            zmean = np.median(Particles.get_pos()[2,:])

        if(mode == 'mean'):
            xmean = np.mean(Particles.get_pos()[0,:])
            ymean = np.mean(Particles.get_pos()[1,:])
            zmean = np.mean(Particles.get_pos()[2,:])

        r = np.sqrt((xmax-xmin)**2+(ymax-ymin)**2+(zmax-zmin)**2)

        self.__params = {'x':xmean,'y':ymean,'z':zmean,'r':r,
                         't':0,'p':0,'zoom':1,
                         'xsize':500, 'ysize':500}
