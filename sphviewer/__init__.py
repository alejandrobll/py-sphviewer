from Particles import Particles
from Camera import Camera
from Scene import Scene
from Render import Render
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.random.rand(1000)
    y = np.random.rand(1000)
    z = np.random.rand(1000)

    hsml = np.random.rand(1000)
    mass = np.random.rand(1000)

    P = Particles(x,y,z,mass,verbose=True)

    S = Scene(P)
    
    plt.ion()
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)

#   S.update_camera(r=10,zoom=1)
#   I = Render(S)
#   ax1.imshow(I.get_image(), extent=I.Scene.get_extent())
#   plt.show()
    for i in xrange(1000):
        S.update_camera(p=360.0*i/999, r=5-(5-0.5)*i/999.,xsize=300,ysize=300)
        image = Render(S)
        print i
        image.save('output/'+str('%04d'% i)+'.png')
#        ax1.imshow(np.log10(image.get_image()+1))
#        plt.draw()
#        ax1.cla()

#    ax1  = fig.add_subplot(221)
#    ax2  = fig.add_subplot(222)
#    ax3  = fig.add_subplot(223)
#    ax4  = fig.add_subplot(224)
#    P.plot('xy', axis=ax1)
#    P.plot('xz', axis=ax2)
#    P.plot('yz', axis=ax3)


 #   print S.Camera._get_camera('xy')

#    plt.show()
#    for i in xrange(100):
#       S.update_camera(t=360.*i/99.,r=0.5)
#       S.Camera.plot('xy',axis=ax1)
#       S.Camera.plot('xz',axis=ax2)
#       S.Camera.plot('yz',axis=ax3)
#       plt.draw()
#       for j in xrange(2):
#           ax1.lines.pop()
#           ax2.lines.pop()
#           ax3.lines.pop()
#       print i
#       #raw_input()
