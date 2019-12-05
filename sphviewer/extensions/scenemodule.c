/*////////////////////////////////////////////////////////////////////
    This file is part of Py-SPHViewer
    
    <Py-SPHVIewer is a framework for rendering particles in Python
    using the SPH interpolation scheme.>
    Copyright (C) <2013>  <Alejandro Benitez Llambay>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
/////////////////////////////////////////////////////////////////////////*/

#include <Python.h>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

#include "projections.h";

void rx(float angle, float *x, float *y, float *z, int n){
  // counter-clockwise rotation matrix along x-axis  
  int i;
  float ytemp, ztemp;
#pragma omp parallel for firstprivate(n,ytemp,ztemp,angle)
  for(i=0;i<n;i++){
    ytemp = y[i];
    ztemp = z[i];
    y[i] = ytemp*cos(angle)+ztemp*sin(angle);
    z[i] = -ytemp*sin(angle)+ztemp*cos(angle);
  }
  return;
}

void ry(float angle, float *x, float *y, float *z, int n){
  // counter-clockwise rotation matrix along y-axis
  int i;
  float xtemp, ztemp;
#pragma omp parallel for firstprivate(n,xtemp,ztemp,angle)
  for(i=0;i<n;i++){
    xtemp = x[i];
    ztemp = z[i];
    x[i] = xtemp*cos(angle)-ztemp*sin(angle);
    z[i] = xtemp*sin(angle)+ztemp*cos(angle);
  }
  return;
}

void rz(float angle, float *x, float *y, float *z, int n){
  // counter-clockwise rotation matrix along z-axis
  int i;
  float xtemp, ytemp;
#pragma omp parallel for firstprivate(n,xtemp,ytemp,angle)
  for(i=0;i<n;i++){
    xtemp = x[i];
    ytemp = y[i];
    x[i] = xtemp*cos(angle)+ytemp*sin(angle);
    y[i] = -xtemp*sin(angle)+ytemp*cos(angle);
  }
  return;
}

float *get_float_array(PyArrayObject *array_obj, int n){
  /* This function returns the data stored in a float PyArrayObject*/

  /*We enfore C contiguous arrays*/
  PyArrayObject *cont_obj = PyArray_ContiguousFromObject(array_obj, PyArray_FLOAT, 1, 3);

  float *local_array = (float *)cont_obj->data;  
  float *output = (float *)malloc( n * sizeof(float) );

#pragma omp parallel for firstprivate(n)
  for(int i=0;i<n;i++){
    output[i] = local_array[i];
  }

  /* release memory */
  Py_DECREF(cont_obj);
  return output;
}

float *get_double_array(PyArrayObject *array_obj, int n){
  /* This function returns the data stored in a double PyArrayObject*/

  /*We enfore C contiguous arrays*/
  PyArrayObject *cont_obj = PyArray_ContiguousFromObject(array_obj, PyArray_DOUBLE, 1, 3);

  double *local_array = (double *)cont_obj->data;  
  float *output = (float *)malloc( n * sizeof(float) );

#pragma omp parallel for firstprivate(n)
  for(int i=0;i<n;i++){
    output[i] = local_array[i];
  }

  /* release memory */
  Py_DECREF(cont_obj);
  return output;
}


long int compute_scene(float *x, float *y, float *z,
		       float *hsml, float *hsml_x, float *hsml_y,
		       float *extent, int xsize, int ysize,
		       float x0_cam, float y0_cam, float z0_cam,
		       int n, long int *kview, float r, float t, 
		       float p, float roll, float zoom, int projection){

  int i;
  float lbin;
  long int idx = 0;
  float xmin, xmax, ymin, ymax;

  // Let's refer the particles to the camera point of view.
  for(i=0;i<n;i++){
    x[i] -= x0_cam;
    y[i] -= y0_cam;
    z[i] -= z0_cam;
  }

  // Let's rotate the particles according to the given angles.
  if(t != 0) rx(t*3.141592/180.0, x, y, z, n);
  if(p != 0) ry(p*3.141592/180.0, x, y, z, n);
  if(roll != 0) rz(roll*3.141592/180.0, x, y, z, n);

  /* Use otographic projection */
  if(projection == 0){
    idx = ortographic_projection(x, y, z, hsml, hsml_x, hsml_y, kview, n,
				 extent, xsize, ysize);
  }
  
  /* Use oblique projection */
  if(projection == 1){
    idx = oblique_projection(x, y, z, hsml, hsml_x, hsml_y, kview, n,
			     r, zoom, extent, xsize, ysize);
  }

  /* Use the equirectangular projection */
  if(projection == 2){
    idx = equirectangular_projection(x, y, z, hsml, hsml_x, hsml_y, kview, n,
				     r, zoom, extent, xsize, ysize);
  } 

  return idx; 
}


static PyObject *scenemodule(PyObject *self, PyObject *args){

  PyArrayObject *x_obj, *y_obj, *z_obj, *h_obj, *h_obj_x, *h_obj_y, *extent_obj;
  float *xlocal, *ylocal, *zlocal, *hlocal, *hlocal_x, *hlocal_y, *extent;
  int n;
  float x0_cam, y0_cam, z0_cam;
  float r,t,p,roll,zoom;
  int xsize, ysize;
  int projection;
  int i;

  if(!PyArg_ParseTuple(args, "OOOOOOffffffffOiii",
		       &x_obj, &y_obj, &z_obj,
		       &h_obj, &h_obj_x, &h_obj_y, 
		       &x0_cam, &y0_cam, &z0_cam, 
		       &r, &t, &p, &roll, &zoom, &extent_obj, 
		       &xsize, &ysize, &projection))
    return NULL;

  n = (int) h_obj_x->dimensions[0];
  
  float * (* get_array) (PyArrayObject *, int); 

  /* check positions data type */
  int type = PyArray_TYPE(x_obj);
  if(type == NPY_FLOAT){
    get_array = get_float_array;
  }
  else if(type == NPY_DOUBLE){
    get_array = get_double_array;
  }else {
    return NULL;
  }

  xlocal = get_array(x_obj, n);
  ylocal = get_array(y_obj, n);
  zlocal = get_array(z_obj, n);

  /* check smoothing data type */
  type = PyArray_TYPE(h_obj_x);
  if(type == NPY_FLOAT){
    get_array = get_float_array;
  }
  else if(type == NPY_DOUBLE){
    get_array = get_double_array;
  }else {
    return NULL;
  }

  hlocal   = get_array(h_obj, n);
  hlocal_x = get_array(h_obj_x, n);
  hlocal_y = get_array(h_obj_y, n);

  extent = (float *) extent_obj->data;

  // Let's do the job
  long int *klocal = (long int *)malloc( n * sizeof(long int) );

  //projection == 0 means r='infinity'
  long int idx = compute_scene(xlocal, ylocal, zlocal,
			       hlocal, hlocal_x, hlocal_y,
			       extent, xsize, ysize,
			       x0_cam, y0_cam, z0_cam,
			       n, klocal, r, t, p, roll,
			       zoom,projection);

  float *x_out_c    = (float *)malloc( idx * sizeof(float) );
  float *y_out_c    = (float *)malloc( idx * sizeof(float) );
  float *h_out_c    = (float *)malloc( idx * sizeof(float) );
  float *h_out_c_x  = (float *)malloc( idx * sizeof(float) );
  float *h_out_c_y  = (float *)malloc( idx * sizeof(float) );
  long int *k_out_c = (long int*)malloc( idx * sizeof(long int) );

#pragma omp parallel for firstprivate(n,idx)
  for(i=0;i<idx;i++){
    x_out_c[i] = xlocal[klocal[i]];
    y_out_c[i] = ylocal[klocal[i]];
    h_out_c[i] = hlocal[klocal[i]];
    h_out_c_x[i] = hlocal_x[klocal[i]];
    h_out_c_y[i] = hlocal_y[klocal[i]];
    k_out_c[i] = klocal[i];
  }

  //I free the local arrays
  free(xlocal);
  free(ylocal);
  free(zlocal);
  free(hlocal);
  free(hlocal_x);
  free(hlocal_y);
  free(klocal);

  // Let's build a numpy array                         
  npy_intp dims[1] = {idx};

  PyArrayObject *x_out = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, x_out_c);
  x_out->flags = NPY_OWNDATA;

  PyArrayObject *y_out = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, y_out_c);
  y_out->flags = NPY_OWNDATA;

  PyArrayObject *h_out = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, h_out_c);
  h_out->flags = NPY_OWNDATA;
  
  PyArrayObject *h_out_x = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, h_out_c_x);
  h_out_x->flags = NPY_OWNDATA;

  PyArrayObject *h_out_y = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, h_out_c_y);
  h_out_y->flags = NPY_OWNDATA;

  PyArrayObject *k_out = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_INT64, k_out_c);
  k_out->flags = NPY_OWNDATA;

  //"NNNN" is the same that OOOO but it does not increase the count
  //reference, allowing the garbage collector to clean up the variables
  //If I use "OOOO" instead, I have to call Py_DECREF(x_out) 
  //Py_DECREAF(y_out) PY_DECREAF(h_out) and Py_DECREAF(k_out)
  PyObject *out = Py_BuildValue("NNNNNN", x_out, y_out, h_out, h_out_x, h_out_y, k_out);

  return out;
}

static PyMethodDef SceneMethods[] = {
  {"scene", scenemodule, METH_VARARGS, "Method for computing the scene"},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "scene",        /* m_name */
    NULL,           /* m_doc */
    -1,             /* m_size */
    SceneMethods,   /* m_methods */
    NULL,           /* m_reload */
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC
PyInit_scene(void)
{
    PyObject *m = PyModule_Create(&moduledef);
    import_array();
    return m;
}
#else
PyMODINIT_FUNC initscene(void) {
  (void) Py_InitModule("scene", SceneMethods);
  import_array();
}
#endif

