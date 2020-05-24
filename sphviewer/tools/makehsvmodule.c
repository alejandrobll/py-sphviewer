//This file is part of Py-SPHViewer

//<Py-SPHVIewer is a framework for rendering particles in Python
//using the SPH interpolation scheme.>
//Copyright (C) <2013>  <Alejandro Benitez Llambay>

//This program is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.

//You should have received a copy of the GNU General Public License
//along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <Python.h>

//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void hsv_to_rgb(float h, float s, float v,
		 float *r, float *g, float *b)
//Function to convert from hsv to rgb color space.
//Coded by Alejandro Benitez-Llambay as a tool for py-sphviewer. The implementation 
//is the same as in colorsys library of python.
{
  if(s == 0.0){
    *r = *g = *b = v;
    return;
  }
  int i = h*6.0; 
  float f = (h*6.0) - i;
  float p = v*(1.0 - s);
  float q = v*(1.0 - s*f);
  float t = v*(1.0 - s*(1.0-f));
  i = i%6;

  if (i == 0){
    *r = v;
    *g = t;
    *b = p;
    return;
  }
  if(i == 1){
    *r = q;
    *g = v;
    *b = p;
    return;
  }
  if(i == 2){
    *r = p;
    *g = v;
    *b = t;
  }
  if(i == 3){
    *r = p;
    *g = q;
    *b = v;
    return;
  }
  if (i == 4){
    *r = t;
    *g = p;
    *b = v;
    return;
  }
  if (i == 5){
    *r = v;
    *g = p;
    *b = q;
    return;
  }
}

void make_image(float *image_h,
		float *image_v,
		float *image_r,
		float *image_g,
		float *image_b,
		int xsize, int ysize,
		float image_h_min, float image_h_max,
		float image_v_min, float image_v_max,
		float huemin, float huemax){

  float h, s, v;
  float r, g, b;
  int i, j;

  
  for(i=0;i<xsize;i++){
    for(j=0;j<ysize;j++){
      h = image_h[j*xsize+i];
      v = image_v[j*xsize+i];

      //Now we clip the values to the desired range and normalize:
      if(h >= image_h_max) h = image_h_max;
      if(h <= image_h_min) h = image_h_min;
      h = huemin+(h-image_h_min)/(image_h_max-image_h_min)*(huemax-huemin);

      if(v >= image_v_max) v = image_v_max;
      if(v <= image_v_min) v = image_v_min;
      v = (v-image_v_min)/(image_v_max-image_v_min);

      s = 1.0 - v; 

      hsv_to_rgb(h,s,v,&r, &g, &b);
      image_r[j*xsize+i] = r;
      image_g[j*xsize+i] = g;
      image_b[j*xsize+i] = b;
    }
  }
}


static PyObject *makehsvmodule(PyObject *self, PyObject *args){
  PyArrayObject *img_h_obj, *img_s_obj, *img_v_obj;
  float *img_h, *img_s, *img_v;
  float *img_r, *img_g, *img_b;
  float img_hmin, img_hmax, img_vmin, img_vmax;
  float hmin, hmax;

  if(!PyArg_ParseTuple(args, "OOffffff",
		       &img_h_obj,
		       &img_v_obj,
		       &img_hmin,
		       &img_hmax,
		       &img_vmin,
		       &img_vmax,
		       &hmin,
		       &hmax))
    return NULL;
    
  // Let's point the C arrays to the numpy arrays
  img_h = (float *)img_h_obj->data;
  img_s = (float *)img_s_obj->data;
  img_v = (float *)img_v_obj->data;

  int xsize = (int *)img_h_obj->dimensions[1];
  int ysize = (int *)img_h_obj->dimensions[0];

  img_r = (float *)malloc( xsize * ysize * sizeof(float) );
  img_g = (float *)malloc( xsize * ysize * sizeof(float) );
  img_b = (float *)malloc( xsize * ysize * sizeof(float) );

  
  // Here we do the work
  make_image(img_h, img_v,
	     img_r, img_g, img_b,
	     xsize, ysize,
	     img_hmin, img_hmax,
	     img_vmin, img_vmax,
	     hmin, hmax);

  // Let's build a numpy array
  npy_intp dims[1] = {xsize*ysize};
  PyArrayObject *image_r_obj = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, img_r);
  image_r_obj->flags = NPY_OWNDATA;
  PyArrayObject *image_g_obj = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, img_g);
  image_g_obj->flags = NPY_OWNDATA;
  PyArrayObject *image_b_obj = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, img_b);
  image_b_obj->flags = NPY_OWNDATA;
  
  return Py_BuildValue("NNN", image_r_obj, 
		       image_g_obj,
		       image_b_obj);
		       
}

static PyMethodDef MakehsvMethods[] = {
  {"makehsv", makehsvmodule, METH_VARARGS, "Method for making a nice HSV image."},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "makehsv",      /* m_name */
    NULL,           /* m_doc */
    -1,             /* m_size */
    MakehsvMethods, /* m_methods */
    NULL,           /* m_reload */
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC
PyInit_makehsv(void)
{
    PyObject *m = PyModule_Create(&moduledef);
    import_array();
    return m;
}
#else
PyMODINIT_FUNC initmakehsv(void) {
  (void) Py_InitModule("makehsv", MakehsvMethods);
  import_array();
}
#endif
