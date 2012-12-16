import numpy as np
import matplotlib.pyplot as plt
from sphviewer import scene
from matplotlib.widgets import Slider, Button

"""
First prototype of an interactive widgets for sphviewer.
For a simple test: python plotter.py
"""

class graph(scene):
    def __init__(self, pos, hsml=None, rho=None, 
                 nb=4, ac=True, cmap='gray',verb=False, res=200):

        scene.__init__(self,pos=pos, hsml=hsml, rho=rho, 
                       nb=nb, ac=ac,verb=verb)

        self.res = res
        self.dens, self.bins, self.extent = \
            self.make_scene(near=True)
        self.cmap = cmap
        self.make_widget()
        self.press = None

    def make_widget(self):
        def update_lim(val):
            vmin = s_vmin.val
            vmax = s_vmax.val
            if vmin>vmax:
                vmin = vmax*(1.0-0.01)
                vmax = vmax
                s_vmin.set_val(vmin)
                s_vmax.set_val(vmax)
            self.image.set_clim(vmin,vmax)
            self.fig.show()

        def update_theta(val):
            self.theta = val
            self.dens, self.bins, self.extent = \
                self.make_scene(near=True)
            self.image.set_array(self.dens)
            self.fig.show()

        def update_phi(val):
            self.phi = val
            self.dens, self.bins, self.extent = \
                self.make_scene(near=True)
            self.image.set_array(self.dens)
            self.fig.show()
        def update_r(val):
            self.r = val
            self.dens, self.bins, self.extent = \
                self.make_scene(near=True)
            self.image.set_array(self.dens)
            self.fig.show()
        def update_zoom(val):
            self.zoom = val
            self.dens, self.bins, self.extent = \
                self.make_scene(near=True)
            self.image.set_array(self.dens)
            self.fig.show()

        def on_press(event):
            self.press = event.x, event.y

        def on_release(event):
            'on release we reset the press data'
            self.press = None

        def on_motion(event):
            if self.press is None: return
            x0, y0 = self.press
            dthetadot = event.x - x0
            dphidot   = event.y - y0
            self.press = event.x, event.y
            self.theta -= dthetadot
            self.phi += dphidot
#            self.s_theta.set_val(self.theta)
#            self.s_phi.set_val(self.phi)
            self.dens, self.bins, self.extent = \
                self.make_scene(near=True)
            self.image.set_array(self.dens)
            self.fig.show()

        def connect():
            'connect to all the events we need'
            self.cidpress = self.fig.canvas.mpl_connect(
                'button_press_event', on_press)
            self.cidrelease = self.fig.canvas.mpl_connect(
                'button_release_event', on_release)
            self.cidmotion = self.fig.canvas.mpl_connect(
                'motion_notify_event', on_motion)
            
        fig = plt.figure('SPH VIEWER FIGURE')
        self.fig = fig
        fig.add_subplot(111)
        self.ax = plt.gca()

        fig_opt = plt.figure("Options")
        
        slide_vmin  = fig_opt.add_axes([0.25,0.1,0.55,0.02])
        slide_vmax  = fig_opt.add_axes([0.25,0.14,0.55,0.02])
        slide_theta = fig_opt.add_axes([0.25,0.18,0.55,0.02])
        slide_phi   = fig_opt.add_axes([0.25,0.22,0.55,0.02])
        slide_r     = fig_opt.add_axes([0.25,0.26,0.55,0.02])
        slide_zoom   = fig_opt.add_axes([0.25,0.30,0.55,0.02])

        vmin = self.dens.min()
        vmax = self.dens.max()

        self.s_vmin = Slider(slide_vmin,
                        'Zmin',vmin,
                        vmax,
                        valinit=vmin)
        self.s_vmax = Slider(slide_vmax,
                        'Zmax',
                        vmin,
                        vmax,
                        valinit=vmax)
        self.s_theta = Slider(slide_theta,
                        r'$\Theta$',
                        0.0,
                        360.0,
                        valinit=self.theta)
        self.s_phi = Slider(slide_phi,
                        r'$\Phi$',
                        0.0,
                        180.0,
                        valinit=self.phi)
        self.s_r = Slider(slide_r,
                        r'$R$',
                        0.0,
                        self.r*10,
                        valinit=self.r)
        self.s_zoom = Slider(slide_zoom,
                        'Zoom',
                        0.1,
                        3.0,
                        valinit=self.zoom)

        self.s_vmin.on_changed(update_lim)
        self.s_vmax.on_changed(update_lim)
        self.s_theta.on_changed(update_theta)
        self.s_phi.on_changed(update_phi)
        self.s_r.on_changed(update_r)
        self.s_zoom.on_changed(update_zoom)

        connect()

        self.image = self.ax.imshow(self.dens,
                            origin='lower',
                            cmap=self.cmap,
                            extent=self.extent,
                            interpolation='nearest'
                            )

        plt.show()

if __name__ == '__main__':
    pos = np.random.rand(3,1000)
    graph(pos)
