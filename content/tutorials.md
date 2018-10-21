---
title: Tutorials
---

# Tutorials

The tutorials below should give you a rough idea of what is Py-SPHViewer capable to do:

<ul>
   {% for item in site.data.tutorial %}
      <li><a href="{{ item.link }}" alt="{{ item.title }}">{{ item.title }}</a></li>
   {% endfor %}
</ul>

If you prefer to get started with Py-SPHViewer and come back to the tutorials later, please continue reading below:

# Getting started

Getting started with Py-SPHViewer is easy. We show below a minimal example on how to obtain the projected density field of the distribution of dark matter particles in a cosmological simulation. After this example you should check the tutorials, which are useful to get started and understanding how Py-SPHViewer works.

The example below uses [this hdf5 file](https://github.com/alejandrobll/py-sphviewer/raw/master/examples/darkmatter_box.h5py), so please download it before running the code below:

```python
import h5py
from sphviewer.tools import QuickView

with h5py.File('darkmatter_box.h5py','r') as f:
    pdrk = f['PartType1/Coordinates'].value

QuickView(pdrk.T, r='infinity', cmap='heat')
```

which produces the following image:

<p align="center">
   <img src="../assets/img/first_image.png" alt="First image with QuickView">
</p>

High-resolution cosmological simulations might result in more impressive results, such us the one shown in the next video:

<p align="center">
   <a href="https://www.youtube.com/watch?annotation_id=annotation_692472089&feature=iv&src_vid=vqGYURAgYUY&v=4ZIgVbNlDU4" target="_blank"><img src="../assets/img/video_stars.png" alt="First image with QuickView"> </a>
</p>
