from scipy import weave
from scipy.weave import converters
import numpy as np
import matplotlib.pyplot as plt

def import_code(filename):
    #naive function to import c code
    string = ''
    fi = open(filename, 'r').readlines()
    for i in fi:
        string += i
    return string

class Render():
    def __init__(self,Scene):    
        self.Scene = Scene
        x,y,t,mass = Scene.get_scene()
        xsize = Scene.Camera.get_params()['xsize']
        ysize = Scene.Camera.get_params()['ysize']
        self.__image = self.__make_render(x,y,t,mass,xsize,ysize)


    def __make_render(self,x,y,t,mass,xsize,ysize):
        n=int(len(x))
        image = np.zeros([xsize,ysize],dtype=(np.float32))
            # C code for making the images
        code       = import_code('/home/alejandro/new_sphviewer/c_code.c')
        shared     = (['x','y', 'xsize', 'ysize',  
                       't','n', 'mass','image'])
        
        # interpolation kernel
        extra_code = import_code('/home/alejandro/new_sphviewer/extra_code.c')
            
        weave.inline(code,shared,
                     support_code=extra_code,
                     type_converters=converters.blitz,
                     compiler='gcc', 
                     extra_compile_args=[' -O3 -fopenmp'],
                     extra_link_args=['-lgomp'])
        return image 

    def get_image(self):
        return self.__image

    def get_max(self):
        return np.max(self.__image)

    def get_min(self):
        return np.min(self.__image)

    def logscale(self):
        self.__image = np.log10(self.__image+1)

    def save(self,outputfile,**kargs):
        plt.imsave(outputfile, self.__image, **kargs)

