#include <Python.h>

//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


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


void c_render(int *x, int *y, int *t, float *mass, 
	      int xsize, int ysize, int n, float *image){ 
  
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
    float mm;
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
  

  // Let's compute the local image 
  //  for(i=(thread_id*ppt); i<(thread_id+1)*ppt; i++){
  for(l=0;l<ppt;l++){
    i = thread_id+nth*l;
    xx = x[i];
    yy = y[i];
    tt = t[i];
    mm = mass[i];

    if(tt < 1) tt = 1;
    if(tt > size_lim) tt = size_lim;
    
    // Let's compute the convolution with the Kernel
    for(j=-tt; j<tt+1; j++){
      for(k=-tt; k<tt+1; k++){
	if( ( (xx+j) >= 0) && ( (xx+j) < xsize) && ( (yy+k) >=0) && ( (yy+k) < ysize)){
	  local_image[(yy+k)*xsize+(xx+j)] += mm*cubic_kernel(sqrt((float)j*(float)j+(float)k*(float)k), tt);
	}
      }
    }
  }
  
  // Let's compute the image for the remainder particles...
  if((r-thread_id) > 0){
    i  = nth*ppt+thread_id;
    xx = x[i];
    yy = y[i];
    tt = t[i];
    mm = mass[i];
    
    if(tt < 1) tt = 1;
    if(tt > size_lim) tt = size_lim;
    
    for(j=-tt; j<tt+1; j++){
      for(k=-tt; k<tt+1; k++){
	if( ( (xx+j) >= 0) && ( (xx+j) < xsize) && ( (yy+k) >=0) && ( (yy+k) < ysize)){
	  local_image[(yy+k)*xsize+(xx+j)] += mm*cubic_kernel(sqrt((float)j*(float)j+(float)k*(float)k), tt);
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
  int *x, *y, *t;
  float *mass;
  int xsize, ysize, n;
  float *image;
  int i;

  xsize = 1000;
  ysize = 1000;
  n = 10000;

  x = (int *)malloc( n * sizeof(int) ); 
  y = (int *)malloc( n * sizeof(int) ); 
  t = (int *)malloc( n * sizeof(int) ); 
  mass = (float *)malloc( n * sizeof(float) ); 
  image = (float *)malloc( xsize * ysize * sizeof(float) ); 

  srand( time(NULL) );
  
  for(i=0;i<n;i++){
    x[i] = rand() % xsize;
    y[i] = rand() % ysize;
    t[i] = rand() % 50;
    mass[i] = rand() % 499;
  }

  c_render(x,y,t,mass,xsize,ysize,n,image);

  FILE *output;

  output = fopen("image_test.bin","wb");
  fwrite(image, sizeof(float), xsize*ysize, output);   
  fclose(output);
}

//Let's start with Python

static PyObject *rendermodule(PyObject *self, PyObject *args){
  PyArrayObject *x_obj, *y_obj, *t_obj;
  PyArrayObject *m_obj;
  int *x, *y, *t;
  float *mass;
  int xsize, ysize;
  int n;
  float *image;

  if(!PyArg_ParseTuple(args, "OOOOii",&x_obj, &y_obj, &t_obj, &m_obj, &xsize, &ysize))
    return NULL;
    
  // Let's check the size of the 1-dimensions arrays.
  n = (int) x_obj->dimensions[0];

  // Let's point the C arrays to the numpy arrays
  x = (int *)x_obj->data;
  y = (int *)y_obj->data;
  t = (int *)t_obj->data;
  mass = (float *)m_obj->data;

  image = (float *)malloc( xsize * ysize * sizeof(float) );

  int i, j;
  for(i=0;i<xsize;i++){
    for(j=0;j<ysize;j++){
      image[j*xsize+i] = 0.0;
    }
  }

  // Here we do the work
  c_render(x,y,t,mass,xsize,ysize,n,image);
  
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

PyMODINIT_FUNC initrender(void) {
  (void) Py_InitModule("render", RenderMethods);
  import_array();
}


// Uncomment the following lines for doing some test.
//void main()
//{
//  test_C();
//}



