from scipy import weave
from scipy.weave import converters
import numpy as np
import matplotlib.pyplot as plt
import os
import colorsys
import Image

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def hsv_to_rgb(h,s,v):
    f = np.vectorize(colorsys.hsv_to_rgb)
    return f(h,s,v)

def rgb_to_hsv(r,g,b):
    f = np.vectorize(colorsys.rgb_to_hsv)
    return f(r,g,b)

def import_code(filename):
    #naive function to import c code
    string = ''
    fi = open(filename, 'r').readlines()
    for i in fi:
        string += i
    return string

class Fancy():
    def __init__(self,Scene):    
        try:
            class_name = Scene._name
        except AttributeError:
            print "You must use a valid class..."
            return
        if(class_name != 'SCENE'):
            print "You must use a valid class..."
            return

        self.Scene = Scene
        x,y,t,kview = Scene.get_scene()
        xsize = Scene.Camera.get_params()['xsize']
        ysize = Scene.Camera.get_params()['ysize']
        self.__image = self.__make_render(x,y,t,kview,xsize,ysize)

        #lets define some flags
        self.__logscale_flag = False;

    def __make_render(self,x,y,t,kview,xsize,ysize):
        n=int(len(x))
        mass = self.Scene._Particles.get_mass()[kview]
#        mass /= np.max(mass)
        try:
            prop1 = self.Scene._Particles.get_prop1()[kview]
            prop2 = self.Scene._Particles.get_prop2()[kview]
        except:
            print 'Check the properties of your particles...'
            return

        image = np.zeros([ysize,xsize,3],dtype=(np.float32))

            # C code for making the images
        code = import_code(os.path.join(PROJECT_ROOT, '.','c_code_fancy.c'))
        shared     = (['x','y', 'xsize', 'ysize',  
                       't','n', 'mass','prop1','prop2','image'])
        # interpolation kernel

        extra_code = import_code(os.path.join(PROJECT_ROOT, '.','extra_code.c'))            
        weave.inline(code,shared,
                     support_code=extra_code,
                     type_converters=converters.blitz,
                     compiler='gcc',
                     extra_compile_args=[' -O3 -fopenmp'],
                     extra_link_args=['-lgomp'])

        h,s,v = rgb_to_hsv(image[:,:,0],
                           image[:,:,1],
                           image[:,:,2])

        self.__vmin = np.min(v)
        self.__vmax = np.max(v)

        return image

    def get_image(self):
        h,s,v = rgb_to_hsv(self.__image[:,:,0],
                           self.__image[:,:,1],
                           self.__image[:,:,2])
        v = np.clip(v, self.__vmin, self.__vmax)
        v = (v-self.__vmin)/(self.__vmax-self.__vmin)
        xsize = self.Scene.Camera.get_params()['xsize']
        ysize = self.Scene.Camera.get_params()['ysize']
        image = np.zeros([ysize,xsize,3],dtype=np.float32)
        image[:,:,0], image[:,:,1], image[:,:,2] = hsv_to_rgb(h,s,v)
        return Image.fromarray((image*255).astype(np.int8),mode='RGB')

    def get_hsv(self):
        return rgb_to_hsv(self.__image[:,:,0],
                          self.__image[:,:,1],
                          self.__image[:,:,2])

    def set_hsv(self,h,s,v):
        self.__image[:,:,0],self.__image[:,:,1],self.__image[:,:,2] = hsv_to_rgb(h,s,v)
        self.__vmin = np.min(v)
        self.__vmax = np.max(v)

    def get_max(self):
        return self.__vmax

    def get_min(self):
        return self.__vmin

    def set_min(self, vmin):
        self.__vmin = vmin

    def set_max(self, vmax):
        self.__vmax = vmax

    def get_extent():
        return self.Scene.get_extent()

    def set_logscale(self,t=True):
        if(t == self.get_logscale()):
            return
        else:
            if(t):
                h,s,v = rgb_to_hsv(self.__image[:,:,0],
                                   self.__image[:,:,1],
                                   self.__image[:,:,2])
                
                self.__k_temp = np.where(v > 0)
                v[self.__k_temp] = np.log10(v[self.__k_temp])
                self.__vmin_temp = np.min(v[self.__k_temp])
                v[self.__k_temp] = (v[self.__k_temp]-self.__vmin_temp)/(-1.0*self.__vmin_temp)
                self.__image[:,:,0],self.__image[:,:,1],self.__image[:,:,2] = hsv_to_rgb(h,s,v)
                self.__logscale_flag = True;
                self.__vmin = np.min(v)
                self.__vmax = np.max(v)
            else:
                h,s,v = rgb_to_hsv(self.__image[:,:,0],
                                   self.__image[:,:,1],
                                   self.__image[:,:,2])
                v[self.__k_temp] = v[self.__k_temp]*(-1.0*self.__vmin_temp)+self.__vmin_temp
                v[self.__k_temp] = 10**v[self.__k_temp]
                self.__image[:,:,0],self.__image[:,:,1],self.__image[:,:,2] = hsv_to_rgb(h,s,v)
                self.__logscale_flag = False;
                self.__vmin = np.min(v)
                self.__vmax = np.max(v)

    def get_logscale(self):
        return self.__logscale_flag

    def histogram(self,axis=None, **kargs):
        h,s,v = hsv_to_rgb(self.__image[:,:,0],
                           self.__image[:,:,1],
                           self.__image[:,:,2])
        if(axis == None):
            axis = plt.gca()
        axis.hist(np.ravel(v), **kargs)
        
    def save(self,outputfile,**kargs):
        plt.imsave(outputfile, self.__image, **kargs)
