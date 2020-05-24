#This file is part of Py-SPHViewer

#<Py-SPHVIewer is a framework for rendering particles in Python
#using the SPH interpolation scheme.>
#Copyright (C) <2013>  <Alejandro Benitez Llambay>

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.interpolate as interp


def get_snapshots_and_times(time, snaplist, timelist):
    current_snap = np.interp(time, timelist, snaplist)
    lower_snapshot = np.floor(current_snap).astype(np.int32)
    upper_snapshot = np.floor(current_snap).astype(np.int32)+1
    if(upper_snapshot < len(timelist)):
        lower_time = timelist[lower_snapshot]
        upper_time = timelist[upper_snapshot]
    else:
        lower_time = timelist[lower_snapshot-1]
        upper_time = timelist[upper_snapshot-1]
    return lower_snapshot, upper_snapshot, lower_time, upper_time


def same_pass(param, frames):
    while 'same' in param:
        idx = param.index('same')
        param[idx] = param[idx-1]

    while 'pass' in param:
        idx = param.index('pass')
        del param[idx]
        del frames[idx]


def get_camera_trajectory(targets, anchors):
    # Author: Alejandro Benitez-Llambay
    # I wrote this function to get the parameters of the camera
    # that define its trajectory according to a small set of anchors.
    # It is based on a first code written by Adrien Thob, who was
    # inspired by the "Surge Target" plugin for Adobe After Effects.

    keys = ['sim_times', 'id_targets', 't', 'p', 'r', 'zoom', 'extent']
    f_interp = {}
    for key in keys:
        frames = [i for i in anchors['id_frames']]
        same_pass(anchors[key], frames)

        if(key == 'id_targets'):
            xc = []
            yc = []
            zc = []
            for i in anchors[key]:
                xc.append(targets[i][0])
                yc.append(targets[i][1])
                zc.append(targets[i][2])
            f_interp['x'] = interp.interp1d(frames, xc)
            f_interp['y'] = interp.interp1d(frames, yc)
            f_interp['z'] = interp.interp1d(frames, zc)
        else:
            f_interp[key] = interp.interp1d(frames, anchors[key])

    camera_params = []
    frames = np.arange(anchors['id_frames'][0], anchors['id_frames'][-1])
    camera_params = []
    keys = ['id_frames', 'sim_times', 'r', 't',
            'p', 'x', 'y', 'z', 'zoom', 'extent']
    for i in frames:
        params = {}
        for key in keys:
            if(key == 'id_frames'):
                params[key] = i
            elif(key == 'extent'):
                value = float(f_interp[key](i))
                params[key] = [-value, value, -value, value]
            else:
                params[key] = float(f_interp[key](i))
        camera_params.append(params)

    return camera_params


if __name__ == "__main__":
    import sphviewer as sph
    import matplotlib.pyplot as plt

    cm_1 = [0.5, 1.5, 0.5]
    cm_2 = [0.5, -5.5, 0.5]

    targets = [cm_1, cm_2]

    anchors = {}
    anchors['sim_times'] = [0.0, 1.0, 'pass', 3.0, 'same', 'same', 'same']
    anchors['id_frames'] = [0, 180, 750, 840, 930, 1500, 1680]
    anchors['r'] = [10, 2, 'same', 4, 2, 'same', 10]
    anchors['id_targets'] = [0, 1, 'same', 'pass', 0, 'same', 1]
    anchors['t'] = [0, 'pass', 'pass', 45, 'pass', 'pass', 0]
    anchors['p'] = [0, 'pass', 'pass', 'pass', 'pass', 'pass', 900]
    anchors['zoom'] = [1., 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['extent'] = [10, 'pass', 'pass', 'pass', 'pass', 'pass', 30]

    data = get_camera_trajectory(targets, anchors)

    n1 = 10000

    cube1 = np.random.rand(3, n1)
    cube1[1, :] -= 6
    cube2 = np.random.rand(3, n1)
    cube2[1, :] += 1
    cubes = np.concatenate((cube1, cube2), axis=1)
    mass = np.ones(n1+n1)

    P = sph.Particles(cubes, mass)
    S = sph.Scene(P)

    h = 0
    for i in data:
        i['xsize'] = 200
        i['ysize'] = 200
        i['roll'] = 0
        S = sph.Scene(P)
        S.update_camera(**i)
        print(S.Camera.get_params())
        R = sph.Render(S)
        img = R.get_image()
        R.set_logscale()
        plt.imsave('test/image_'+str('%d.png' % h), img,
                   vmin=0, vmax=6, cmap='cubehelix')
        h += 1
