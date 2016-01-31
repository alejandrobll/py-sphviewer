//#include <Python.h>

//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

//#include <numpy/ndarraytypes.h>
//#include <numpy/ndarrayobject.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

float hsv_to_rgb(float h, float s, float v,
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
		float *image_s,
		float *image_v,
		float *image_r,
		float *image_g,
		float *image_b,
		int xsize, int ysize){

  float h, s, v;
  float r, g, b;
  int i, j;

  
  for(i=0;i<xsize;i++){
    for(j=0;j<ysize;j++){
      h = image_h[j*xsize+i];
      s = image_s[j*xsize+i];
      v = image_v[j*xsize+i];
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
  int xsize, ysize;

  if(!PyArg_ParseTuple(args, "OOOii",
		       &img_h_obj,
		       &img_s_obj,
		       &img_v_obj, 
		       &xsize, 
		       &ysize))
    return NULL;
    
  // Let's point the C arrays to the numpy arrays
  img_h = (float *)img_h_obj->data;
  img_s = (float *)img_s_obj->data;
  img_v = (float *)img_v_obj->data;

  img_r = (float *)malloc( xsize * ysize * sizeof(float) );
  img_g = (float *)malloc( xsize * ysize * sizeof(float) );
  img_b = (float *)malloc( xsize * ysize * sizeof(float) );

  
  // Here we do the work
  make_image(img_h, img_s, img_v,
	     img_r, img_g, img_b,
	     xsize, ysize);

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

static PyMethodDef HSVMethods[] = {
  {"render", makehsvmodule, METH_VARARGS, "Method for making a nice HSV image."},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initmakehsv(void) {
  (void) Py_InitModule("make_hsv", HSVMethods);
  import_array();
}






//void main(){
//  float h, s, v;
//  float r, g, b;
//  float rr, gg, bb;
//
//  h = 0.1;
//  s = 0.5;
//  v = 0.3;
//
//  hsv_to_rgb(h,s,v,&r, &g, &b);
//
//  printf("%.2f\t%.2f\t%.2f\n", r,g,b);
//
//}
