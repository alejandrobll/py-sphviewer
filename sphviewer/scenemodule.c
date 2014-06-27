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

    lbin = 2*xmax/xsize;

    for(i=0;i<n;i++){
      if( (x[i] >= xmin) & (x[i] <= xmax) & 
	  (y[i] >= ymin) & (y[i] <= ymax) ) {
	
	kview[idx] = i;
	idx += 1;
      }
    }

#pragma omp parallel for firstprivate(n,xmin,xmax,ymin,ymax,xsize,ysize,lbin)
    for(i=0;i<n;i++){
      x[i] = (x[i] - xmin) / (xmax-xmin) * xsize;
      y[i] = (y[i] - ymin) / (ymax-ymin) * ysize;
      hsml[i] = hsml[i]/lbin;
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

      lbin = 2*xmax/xsize;
      
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
	hsml[i] = (hsml[i]/zpart)/lbin;
      }
    }          
  return idx;
}


static PyObject *scenemodule(PyObject *self, PyObject *args){

  PyArrayObject *x_obj, *y_obj, *z_obj, *h_obj, *extent_obj;
  float *x, *y, *z, *h;
  int n;
  float *extent;
  float x0_cam, y0_cam, z0_cam;
  float r,t,p,roll,zoom;
  int xsize, ysize;
  int projection;

  if(!PyArg_ParseTuple(args, "OOOOffffffffOiii",
		       &x_obj, &y_obj, &z_obj, &h_obj, 
		       &x0_cam, &y0_cam, &z0_cam, 
		       &r, &t, &p, &roll, &zoom, &extent_obj, 
		       &xsize, &ysize, &projection))
    return NULL;

  n = (int) x_obj->dimensions[0];

  // Let's point to the data of the objects
  x = (float *)x_obj->data;
  y = (float *)y_obj->data;
  z = (float *)z_obj->data;
  h = (float *)h_obj->data;
  extent = (float *)extent_obj->data;

  // Let's make a local copy of the variables, just to preserve the input values
  float *xlocal, *ylocal, *zlocal, *hlocal;
  xlocal = (float *)malloc( n * sizeof(float) );
  ylocal = (float *)malloc( n * sizeof(float) );
  zlocal = (float *)malloc( n * sizeof(float) );
  hlocal = (float *)malloc( n * sizeof(float) );

  int i;
#pragma omp parallel for firstprivate(n)
  for(i=0;i<n;i++){
    xlocal[i] = x[i];
    ylocal[i] = y[i];
    zlocal[i] = z[i];
    hlocal[i] = h[i];
  }

  // Let's do the job
  long int *klocal;
  klocal = (long int *)malloc( n * sizeof(long int) );

  //projection == 0 means r='infinity'
  long int idx = compute_scene(xlocal, ylocal, zlocal, hlocal, 
			       extent, xsize, ysize,
			       x0_cam, y0_cam, z0_cam,
			       n, klocal, r, t, p, roll,
			       zoom,projection);

  float *x_out_c, *y_out_c, *h_out_c;
  long int *k_out_c;

  x_out_c = (float *)malloc( idx * sizeof(float) );
  y_out_c = (float *)malloc( idx * sizeof(float) );
  h_out_c = (float *)malloc( idx * sizeof(float) );
  k_out_c = (long int*)malloc( idx * sizeof(long int) );

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

PyMODINIT_FUNC initscene(void) {
  (void) Py_InitModule("scene", SceneMethods);
  import_array();
}

