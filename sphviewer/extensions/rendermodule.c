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

float *get_double_array(PyArrayObject *array_obj, int n){
  /* This function returns the data stored in a double PyArrayObject*/
  double *local_array = (double *)array_obj->data;
  float *output = (float *)malloc( n * sizeof(float) );

#pragma omp parallel for firstprivate(n)
  for(int i=0;i<n;i++){
    output[i] = local_array[i];
  }

  return output;
}


float cubic_kernel(float r, float h){
  //2D Dome-shapeed quadratic Kernel (1-R^2) (Hicks and Liebrock 2000).
  float func;
  float sigma;

  sigma = 15.0/(8.0*3.141592);
  if(r/h <= 1.0)
    func = 4.0/3.0*h*pow(sqrt(1.0-(r/h)*(r/h)),3);
  if(r/h > 1.0)
    func = 0;
  return sigma*func/(h*h*h);
}

float sinc(float x){
  float value;

  if(x==0)
    value = 1.0;
  else
    value = sin(x)/x;
  return value;
}


void c_render(float *x, float *y, float *t, float *mass,
	      int xsize, int ysize, int n, int projection, float *extent, float *image){

  // C function calculating the image of the particles convolved with our kernel
  int size_lim;

  if(xsize >= ysize){
    size_lim = xsize;
  }
  else{
    size_lim = ysize;
  }

#pragma omp parallel
  {
    float *local_image;
    int i,j,k,l;
    int xx, yy, tt;
    float tt_f, mm;
    int r, nth, ppt, thread_id;

  nth = omp_get_num_threads();                // Get number of threads
  thread_id = omp_get_thread_num();           // Get thread id
  ppt = n/nth;                                //  Number or Particles per thread (ppt)
  r = n-ppt*nth;                              // Remainder


  local_image = (float *)malloc( xsize * ysize * sizeof(float) );

  // Let's initialize the image to zero

  for(j=0;j<xsize;j++){
    for(k=0;k<ysize;k++){
      local_image[k*xsize+j] = 0.0;
    }
  }
  // Dome projection/Fish-eye camera : Azimuth Equidistant projection distortions
  if(projection==2){
    float xxrad, yyrad, ttrad, rho, sincrho, cosrho;
    if((r-thread_id) > 0) ppt+=1; //to include the remainder particle
    // Let's compute the local image
    //  for(i=(thread_id*ppt); i<(thread_id+1)*ppt; i++){
    for(l=0;l<ppt;l++){
      i = thread_id+nth*l;
      xx = (int) x[i];
      yy = (int) y[i];
      xxrad = (((float)xx/(float)xsize)*(extent[1]-extent[0])+extent[0])*3.141592;
      yyrad = (((float)yy/(float)ysize)*(extent[3]-extent[2])+extent[2])*3.141592;
      rho = sqrt(xxrad*xxrad+yyrad*yyrad);
      sincrho = sinc(rho);
      cosrho = cos(rho);
      tt_f = t[i];
      ttrad = ((tt_f/(float)xsize)*(extent[1]-extent[0]))*3.141592;
      tt = (int)ceil(tt_f/sincrho);
      mm = mass[i];

      if(tt <= 1) {
        local_image[yy*xsize+xx] += mm;
        continue;
      }
      if(tt > size_lim) tt = size_lim;

      int jxx,kyy;
      float jxxrad, kyyrad, rhopix, cosd;
      // Let's compute the convolution with the Kernel
      for(j=-tt; j<tt+1; j++){
        for(k=-tt; k<tt+1; k++){
      jxx = j+xx;
      kyy = k+yy;
      jxxrad = ((((float)jxx)/(float)xsize)*(extent[1]-extent[0])+extent[0])*3.141592;
      kyyrad = ((((float)kyy)/(float)ysize)*(extent[3]-extent[2])+extent[2])*3.141592;
      rhopix = sqrt(jxxrad*jxxrad+kyyrad*kyyrad);
  	if( (rhopix <= 3.141592) && (jxx>=0) && (jxx<xsize) && (kyy>=0) && (kyy<ysize) ){
        cosd = (xxrad*jxxrad+yyrad*kyyrad)*sincrho*sinc(rhopix)+cosrho*cos(rhopix);
      if(cosd>=1) cosd = 1; else if(cosd<=-1) cosd = -1;
  	  local_image[kyy*xsize+jxx] += mm*cubic_kernel(acos(cosd), ttrad); //I should normalize here by the surface area
  	}
        }
              }
            }

          }
          //
          else{
            if((r-thread_id) > 0) ppt+=1; //to include the remainder particle
            // Let's compute the local image
            //  for(i=(thread_id*ppt); i<(thread_id+1)*ppt; i++){
            for(l=0;l<ppt;l++){
              i = thread_id+nth*l;
              xx = (int) x[i];
              yy = (int) y[i];
              tt_f = t[i];
              tt = (int)ceil(tt_f);
              mm = mass[i];

              if(tt <= 1) {
                local_image[yy*xsize+xx] += mm;
                continue;
              }
              if(tt > size_lim) tt = size_lim;

              // Let's compute the convolution with the Kernel
              for(j=-tt; j<tt+1; j++){
                for(k=-tt; k<tt+1; k++){
          	if( ( (xx+j) >= 0) && ( (xx+j) < xsize) && ( (yy+k) >=0) && ( (yy+k) < ysize)){
          	  local_image[(yy+k)*xsize+(xx+j)] += mm*cubic_kernel(sqrt((float)j*(float)j+(float)k*(float)k), tt_f);
          	}
                }
      }
    }
  }
  // Let's merge the local images


  #pragma omp critical
  {
    for(j=0;j<xsize;j++){
      for(k=0;k<ysize;k++){
	image[k*xsize+j] += local_image[k*xsize+j];
      }
    }
    free(local_image);
  }
  }
  return;
}


void test_C(){
  // This function if for testing purposes only. It writes a file called image_test.bin
  int *x, *y;
  float *t, *mass;
  int xsize, ysize, n, projection;
  float *extent, *image;
  int i;

  xsize = 1000;
  ysize = 1000;
  n = 10000;

  x = (int *)malloc( n * sizeof(int) );
  y = (int *)malloc( n * sizeof(int) );
  t = (float *)malloc( n * sizeof(int) );
  mass = (float *)malloc( n * sizeof(float) );
  image = (float *)malloc( 4 * sizeof(float) );
  image = (float *)malloc( xsize * ysize * sizeof(float) );

  srand( time(NULL) );

  for(i=0;i<n;i++){
    x[i] = rand() % xsize;
    y[i] = rand() % ysize;
    t[i] = rand() % 50;
    mass[i] = rand() % 499;
  }

  c_render(x,y,t,mass,xsize,ysize,n,projection,extent,image);

  FILE *output;

  output = fopen("image_test.bin","wb");
  fwrite(image, sizeof(float), xsize*ysize, output);
  fclose(output);
}

//Let's start with Python

static PyObject *rendermodule(PyObject *self, PyObject *args){
  PyArrayObject *x_obj, *y_obj, *t_obj;
  PyArrayObject *m_obj, *extent_obj;
  float *x, *y, *t;
  float *mass;
  int xsize, ysize;
  int n, projection;
  float *extent, *image;
  int DOUBLE = 0;

  if(!PyArg_ParseTuple(args, "OOOOiiiO",&x_obj, &y_obj, &t_obj, &m_obj, &xsize, &ysize, &projection, &extent_obj))
    return NULL;

  // Let's check the size of the 1-dimensions arrays.
  n = (int) m_obj->dimensions[0];

  // Let's point the C arrays to the numpy arrays
  x    = (float *)x_obj->data;
  y    = (float *)y_obj->data;
  t    = (float *)t_obj->data; /* These are always floats, as they come from Scene */



  /* Let's check the type of mass, which could be either double or float */
  int type = PyArray_TYPE(m_obj);
  if(type == NPY_FLOAT){
    mass = (float *)m_obj->data;
  }
  else if(type == NPY_DOUBLE){
    mass = get_double_array(m_obj, n);
    DOUBLE = 1;
  }else {
    return NULL;
  }

  extent = (float *)extent_obj->data;

  image = (float *)malloc( xsize * ysize * sizeof(float) );

  int i, j;
  for(i=0;i<xsize;i++){
    for(j=0;j<ysize;j++){
      image[j*xsize+i] = 0.0;
    }
  }

  // Here we do the work
  c_render(x,y,t,mass,xsize,ysize,n,projection,extent,image);

  if(DOUBLE) free(mass);

  // Let's build a numpy array
  npy_intp dims[1] = {xsize*ysize};
  PyArrayObject *image_obj = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, image);
  image_obj->flags = NPY_OWNDATA;

  return Py_BuildValue("N", image_obj);

}

static PyMethodDef RenderMethods[] = {
  {"render", rendermodule, METH_VARARGS, "Method for rendering the image."},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "render",       /* m_name */
    NULL,           /* m_doc */
    -1,             /* m_size */
    RenderMethods,  /* m_methods */
    NULL,           /* m_reload */
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC
PyInit_render(void)
{
    PyObject *m = PyModule_Create(&moduledef);
    import_array();
    return m;
}
#else
PyMODINIT_FUNC initrender(void) {
  (void) Py_InitModule("render", RenderMethods);
  import_array();
}
#endif


// Uncomment the following lines for doing some test.
//void main()
//{
//  test_C();
//}
