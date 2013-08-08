import numpy as np
import matplotlib.pyplot as plt
from Camera import Camera

def rotate(angle, axis, pos):
    angle *= np.pi/180.0
    if axis == 'x':
        R = np.array([[1,0,0],
                      [0,np.cos(angle),np.sin(angle)],
                      [0,-np.sin(angle),np.cos(angle)]])
    elif axis == 'y':
        R = np.array([[np.cos(angle),0,-np.sin(angle)],
                      [0,1,0],
                      [np.sin(angle),0,np.cos(angle)]])
    elif axis == 'z':
        R = np.array([[np.cos(angle),np.sin(angle),0],
                      [-np.sin(angle),np.cos(angle),0],
                      [0,0,1]])
    return np.dot(R,pos)

class Scene():
    def __init__(self, Particles):
        try:
            particles_name = Particles._name
        except AttributeError:
            print "You must use a valid class..."
            return
        if(particles_name != 'PARTICLES'):
            print "You must use a valid class..."
            return

        self._name = 'SCENE'
        self.Camera = Camera()
        self._Particles = Particles
        #I use the autocamera by default
        self.Camera.set_autocamera(Particles)
        self._camera_params = self.Camera.get_params()
        self.__x, self.__y, self.__hsml, self.__kview = self.__compute_scene()

    def set_autocamera(self,mode='density'):
        self.Camera.set_autocamera(self._Particles,mode=mode)
        self._camera_params = self.Camera.get_params()
        self.__x, self.__y, self.__hsml, self.__kview = self.__compute_scene()

    def get_scene(self):
        return self.__x, self.__y, self.__hsml, self.__kview

    def update_camera(self,**kargs):
        self.Camera.set_params(**kargs)
        self.__x, self.__y, self.__hsml, self.__kview = self.__compute_scene()

    def __compute_scene(self):
        
        pos = (1.0*self._Particles.get_pos()).astype(np.float32)
        
        pos[0,:] -= np.array([self._camera_params['x']])
        pos[1,:] -= np.array([self._camera_params['y']])
        pos[2,:] -= np.array([self._camera_params['z']])

        if self._camera_params['t'] != 0:
            pos = rotate(self._camera_params['t'],'x',pos)
        if self._camera_params['p'] != 0:
            pos = rotate(self._camera_params['p'],'y',pos)

        if(self._camera_params['r'] == 'infinity'):                
            xmax = np.max(pos[0,:])
            xmin = np.min(pos[0,:])
            ymax = np.max(pos[1,:])
            ymin = np.min(pos[1,:])

            lmax = max(xmax,ymax)
            lmin = min(ymax,ymin)

            xmin = ymin = lmin
            xmax = ymax = ymax

            self.__extent = np.array([xmin+self._camera_params['x'],
                                      xmax+self._camera_params['x'],
                                      ymin+self._camera_params['y'],
                                      ymax+self._camera_params['y']])

#            self.__extent = np.array([xmin,xmax,ymin,ymax])
            lbin = 2*xmax/self._camera_params['xsize']
                   
            pos[0,:] = (pos[0,:]-xmin)/(xmax-xmin)*self._camera_params['xsize']
            pos[1,:] = (pos[1,:]-ymin)/(ymax-ymin)*self._camera_params['ysize']
            hsml = self._Particles.get_hsml()/lbin
            kview = np.arange(np.size(hsml))

            return pos[0,:], pos[1,:], hsml, kview

        else:
            pos[2,:] -= (-1.0*self._camera_params['r'])
        
            FOV  = 2.*np.abs(np.arctan(1./self._camera_params['zoom']))
        
            xmax = self._camera_params['zoom']*np.tan(FOV/2.)
            xmin = -xmax
            ymax = 0.5*(xmax-xmin)*self._camera_params['ysize']/self._camera_params['xsize']
            ymin = -ymax
            xfovmax =  FOV/2.*180./np.pi
            xfovmin =  -FOV/2.*180./np.pi
            # in order to have symmetric y limits
            yfovmax = 0.5*((xfovmax-xfovmin)*
                           self._camera_params['ysize']/self._camera_params['xsize'])
            yfovmin = -yfovmax
            self.__extent = np.array([xfovmin,xfovmax,yfovmin,yfovmax])
            lbin = 2*xmax/self._camera_params['xsize']
            
            kview, = np.where((pos[2,:] > 0.) & 
                              (np.abs(pos[1,:])<=(np.abs(pos[2,:])* 
                                                  np.tan(FOV/2.))) &
                              (np.abs(pos[1,:]) <= (np.abs(pos[2,:])*
                                                    np.tan(FOV/2.))))
            pos   = pos[:,kview]
            hsml  = self._Particles.get_hsml()[kview]
        
            pos[0,:] = ((pos[0,:]*self._camera_params['zoom']/ 
                         pos[2,:]-xmin)/(xmax-xmin)*
                        (self._camera_params['xsize']-1.))
            pos[1,:] = ((pos[1,:]*self._camera_params['zoom']/ 
                         pos[2,:]-ymin)/(ymax-ymin)*
                        (self._camera_params['ysize']-1.))
            hsml = (hsml*self._camera_params['zoom']/pos[2,:]/lbin)
            
            return pos[0,:], pos[1,:], hsml, kview

        
    def get_extent(self):
        return self.__extent

    def plot(self,axis=None,**kargs):
        if(axis == None):
            axis = plt.gca()
        axis.plot(self.__x, self.__y, 'k.', **kargs)
        
