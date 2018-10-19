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
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


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


long int compute_scene(float *x, float *y, float *z, float *hsml, 
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

  // projection == 0 places the camera at the infinity and uses extent as limits.
  // However, the z-range in the image is preserved.
  if(projection == 0){
    xmin = extent[0];
    xmax = extent[1];
    ymin = extent[2];
    ymax = extent[3];


    for(i=0;i<n;i++){
      if( (x[i] >= xmin) & (x[i] <= xmax) & 
	  (y[i] >= ymin) & (y[i] <= ymax) ) {
	
	kview[idx] = i;
	idx += 1;
      }
    }

#pragma omp parallel for firstprivate(n,xmin,xmax,ymin,ymax,xsize,ysize)
    for(i=0;i<n;i++){
      x[i] = (x[i] - xmin) / (xmax-xmin) * xsize;
      y[i] = (y[i] - ymin) / (ymax-ymin) * ysize;
      hsml[i] = hsml[i]/(xmax-xmin) * xsize;
    }          
  }
  // If the camera is not at the infinity, let's put it at a certain 
  // distance from the object.
  else
    {
      float FOV = 2.0*fabsf(atanf(1.0/zoom));
      xmax = 1.0;
      xmin = -xmax;
      //Let's preserve the pixel aspect ration
      ymax = 0.5*(xmax-xmin)*ysize/xsize;
      ymin = -ymax;
      
      float xfovmax = FOV/2.*180.0/3.141592;
      float xfovmin = -xfovmax;
      float yfovmax = 0.5*(xfovmax-xfovmin)*ysize/xsize;
      float yfovmin = -yfovmax;

      extent[0] = xfovmin;
      extent[1] = xfovmax;
      extent[2] = yfovmin;
      extent[3] = yfovmax;

      float zpart;
      for(i=0;i<n;i++){
	zpart = (z[i]-(-1.0*r));
	if( (zpart > 0) & 
	    (fabsf(x[i]) <= fabsf(zpart)*xmax/zoom) &
	    (fabsf(y[i]) <= fabsf(zpart)*ymax/zoom) ) {
	  
	  kview[idx] = i;
	  idx += 1;
	}
      }

#pragma omp parallel for firstprivate(n,xmin,xmax,ymin,ymax,xsize,ysize,lbin,zoom,r,zpart)
      for(i=0;i<n;i++){
	zpart = (z[i]-(-1.0*r))/zoom;
	x[i] = (x[i]/zpart - xmin) / (xmax-xmin) * xsize;
	y[i] = (y[i]/zpart - ymin) / (ymax-ymin) * ysize;
	hsml[i] = (hsml[i]/zpart)/(xmax-xmin) * xsize;
      }
    }          
  return idx;
}


static PyObject *scenemodule(PyObject *self, PyObject *args){

  PyArrayObject *x_obj, *y_obj, *z_obj, *h_obj, *extent_obj;
  float *xlocal, *ylocal, *zlocal, *hlocal, *extent;
  int n;
  float x0_cam, y0_cam, z0_cam;
  float r,t,p,roll,zoom;
  int xsize, ysize;
  int projection;
  int i;

  if(!PyArg_ParseTuple(args, "OOOOffffffffOiii",
		       &x_obj, &y_obj, &z_obj, &h_obj, 
		       &x0_cam, &y0_cam, &z0_cam, 
		       &r, &t, &p, &roll, &zoom, &extent_obj, 
		       &xsize, &ysize, &projection))
    return NULL;

  n = (int) h_obj->dimensions[0];
  
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

  /* check positions data type */
  type = PyArray_TYPE(h_obj);
  if(type == NPY_FLOAT){
    get_array = get_float_array;
  }
  else if(type == NPY_DOUBLE){
    get_array = get_double_array;
  }else {
    return NULL;
  }

  hlocal = get_array(h_obj, n);


  extent = (float *) extent_obj->data;

  // Let's do the job
  long int *klocal = (long int *)malloc( n * sizeof(long int) );

  //projection == 0 means r='infinity'
  long int idx = compute_scene(xlocal, ylocal, zlocal, hlocal, 
			       extent, xsize, ysize,
			       x0_cam, y0_cam, z0_cam,
			       n, klocal, r, t, p, roll,
			       zoom,projection);

  float *x_out_c    = (float *)malloc( idx * sizeof(float) );
  float *y_out_c    = (float *)malloc( idx * sizeof(float) );
  float *h_out_c    = (float *)malloc( idx * sizeof(float) );
  long int *k_out_c = (long int*)malloc( idx * sizeof(long int) );

#pragma omp parallel for firstprivate(n,idx)
  for(i=0;i<idx;i++){
    x_out_c[i] = xlocal[klocal[i]];
    y_out_c[i] = ylocal[klocal[i]];
    h_out_c[i] = hlocal[klocal[i]];
    k_out_c[i] = klocal[i];
  }

  //I free the local arrays
  free(xlocal);
  free(ylocal);
  free(zlocal);
  free(hlocal);
  free(klocal);

  // Let's build a numpy array                         
  npy_intp dims[1] = {idx};

  PyArrayObject *x_out = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, x_out_c);
  x_out->flags = NPY_OWNDATA;

  PyArrayObject *y_out = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, y_out_c);
  y_out->flags = NPY_OWNDATA;

  PyArrayObject *h_out = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, h_out_c);
  h_out->flags = NPY_OWNDATA;

  PyArrayObject *k_out = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_INT64, k_out_c);
  k_out->flags = NPY_OWNDATA;

  //"NNNN" is the same that OOOO but it does not increase the count
  //reference, allowing the garbage collector to clean up the variables
  //If I use "OOOO" instead, I have to call Py_DECREF(x_out) 
  //Py_DECREAF(y_out) PY_DECREAF(h_out) and Py_DECREAF(k_out)
  PyObject *out = Py_BuildValue("NNNN", x_out, y_out, h_out, k_out);

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

