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

/* Kernel choice */
#include "../sph_kernel.h"

/* Simple definition of min and max */
/**
 * @brief Minimum of two numbers
 *
 * This macro evaluates its arguments exactly once.
 */
#define min(a, b)                                                              \
  ({                                                                           \
    const __typeof__(a) _a = (a);                                              \
    const __typeof__(b) _b = (b);                                              \
    _a < _b ? _a : _b;                                                         \
  })

/**
 * @brief Maximum of two numbers
 *
 * This macro evaluates its arguments exactly once.
 */
#define max(a, b)                                                              \
  ({                                                                           \
    const __typeof__(a) _a = (a);                                              \
    const __typeof__(b) _b = (b);                                              \
    _a > _b ? _a : _b;                                                         \
  })

/**
 * @brief Main rendering function. Takes in arrays cooresponding to SPH
 *        particles, and an image, and (in parallel) smooths the data onto
 *        the grid. All input properties should be in grid units, so if
 *        it's a 128 x 128 grid, x, y, h should all be bounded within [0, 128].
 *
 *
 * @param *x, pointer to the array of x positions
 * @param *y, pointer to the array of y positions
 * @param *t, pointer to the array of smoothing lengths
 * @param *mass, ponter to the array of particle masses
 * @param xsize, ysize, sizes of the image grid
 * @param n, number of particles
 * @param *image, pointer to the image array of size xsize * ysize.
 *
 */
void c_render(float *x, float *y, float *t, float *mass, int xsize, int ysize,
              int n, float *image) {

  /* What's the largest dimension? */
  const float size_lim = (float)(xsize > ysize ? xsize : ysize);

/* Parallel loop; each thread has its own thread-local image that are combined
 * at the end. Note that everything here is done in terms of the pixel grid
 * co-ordinates. */
#pragma omp parallel
  {
    int n_threads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();
    /* Particles per thread, n is total number of particles. */
    const int particles_per_thread = n / n_threads;
    const int remainder_particles = n - particles_per_thread * n_threads;

    /* Allocate the local image that we'll spread our particles onto */
    float *local_image = (float *)malloc(xsize * ysize * sizeof(float));
    bzero(local_image, xsize * ysize * sizeof(float));

    /* Loop over the particles only belonging to our thread */
    for (int i = (thread_id * particles_per_thread);
         i < (thread_id + 1) * particles_per_thread; i++) {

      const float x_float = x[i];
      const float y_float = y[i];
      const int x_cell = (int)x_float;
      const int y_cell = (int)y_float;

      const float particle_mass = mass[i];
      /* Curtail the smoothing length if it is to large */
      const float smoothing_length = min(t[i], size_lim);

      if (smoothing_length < 0.5f) {
        /* If the smoothing length is less than half of a pixels width, we
         * can contribute in a very simplistic way to the density of the pixel
         * which we know has area A = 1. */

        const int pixel = y_cell * xsize + x_cell;
        local_image[pixel] += particle_mass; /* I.e. mass / A with A=1 */

        /* We're done! */
        continue;
      }

      /* Cast the smoothing length to an integer so we can loop over the pixel
       * square cast by it */
      const int pixels_to_loop_over = (int)smoothing_length;

      /* Loop over the pixels that our kernel covers. */
      for (int j = -pixels_to_loop_over; j < pixels_to_loop_over + 1; j++) {
        /* Need to check if we live within the bounds of the image */
        if (((x_cell + j) >= 0) && ((x_cell + j) < xsize)) {
          /* Compute x-only properties that will stay constant for y loop */
          const float distance_x = ((float)x_cell + (float)j + 0.5f) - x_float;
          const float distance_x_2 = distance_x * distance_x;

          for (int k = -pixels_to_loop_over; k < pixels_to_loop_over + 1; k++) {
            if (((y_cell + k) >= 0) && ((y_cell + k) < ysize)) {
              const float distance_y =
                  ((float)y_cell + (float)k + 0.5f) - y_float;
              const float distance_y_2 = distance_y * distance_y;

              const float radius = sqrtf(distance_y_2 + distance_x_2);
              /* Can call the kernel! Woo! */
              const float kernel = cubic_kernel(radius, smoothing_length);

              /* Now add onto the correct cell */
              const int pixel = (y_cell + k) * xsize + (x_cell + j);
              local_image[pixel] += particle_mass * kernel;
            }
          }
        }
      }
    }

    /* Let's compute the image for the remainder particles... */
    if ((remainder - thread_id) > 0) {
      const int i = n_threads * particles_per_thread + thread_id;
      const float x_float = x[i];
      const float y_float = y[i];
      const int x_cell = (int)x_float;
      const int y_cell = (int)y_float;

      const float particle_mass = mass[i];
      /* Curtail the smoothing length if it is to large */
      const float smoothing_length = min(t[i], size_lim);

      if (smoothing_length < 0.5f) {
        /* If the smoothing length is less than half of a pixels width, we
         * can contribute in a very simplistic way to the density of the pixel
         * which we know has area A = 1. */

        const int pixel = y_cell * xsize + x_cell;
        local_image[pixel] += particle_mass; /* I.e. mass / A with A=1 */

        /* We're done! */
      } else {
        /* Gotta do the smoothing as usual */

        /* Cast the smoothing length to an integer so we can loop over the pixel
         * square cast by it */
        const int pixels_to_loop_over = (int)smoothing_length;

        /* Loop over the pixels that our kernel covers. */
        for (int j = -pixels_to_loop_over; j < pixels_to_loop_over + 1; j++) {
          /* Need to check if we live within the bounds of the image */
          if (((x_cell + j) >= 0) && ((x_cell + j) < xsize)) {
            /* Compute x-only properties that will stay constant for y loop */
            /* Distance from particle to center of cell */
            const float distance_x =
                ((float)x_cell + (float)j + 0.5f) - x_float;
            const float distance_x_2 = distance_x * distance_x;

            for (int k = -pixels_to_loop_over; k < pixels_to_loop_over + 1;
                 k++) {
              if (((y_cell + k) >= 0) && ((y_cell + k) < ysize)) {
                /* Distance from particle to center of cell */
                const float distance_y =
                    ((float)y_cell + (float)k + 0.5f) - y_float;
                const float distance_y_2 = distance_y * distance_y;

                const float radius = sqrtf(distance_y_2 + distance_x_2);
                /* Can call the kernel! Woo! */
                const float kernel = cubic_kernel(radius, smoothing_length);

                /* Now add onto the correct cell */
                const int pixel = (y_cell + k) * xsize + (x_cell + j);
                local_image[pixel] += particle_mass * kernel;
              }
            }
          }
        }
      }
    }
#pragma omp critical
    /* Send everyone back home and merge our threadlocal image with the
     * full one in main memory. */
    {
      for (int j = 0; j < xsize; j++) {
        for (int k = 0; k < ysize; k++) {
          image[k * xsize + j] += local_image[k * xsize + j];
        }
      }
      free(local_image);
    }
  }
  return;
}

/* Below this point we have the python interface funuctions */
/* ---------------------- P Y T H O N ----------------------*/

/**
 * @brief: Returns the data sotred in a double PyArrayObject.
 *
 * @param[in] *array_obj, the python array object
 * @param[in] n, size of the object
 * @param[out] output, a pointer to a C array containing the same stuff
 *             as (double precision floating point numbers) as the
 *             python object array
 */
float *get_double_array(PyArrayObject *array_obj, int n) {
  double *local_array = (double *)array_obj->data;
  float *output = (float *)malloc(n * sizeof(float));

#pragma omp parallel for firstprivate(n)
  for (int i = 0; i < n; i++) {
    output[i] = local_array[i];
  }

  return output;
}

/**
 * @brief Main python object that interfaces the c_render function
 *        with the internal python code.
 *
 * @param *self, own python object
 * @param *args, list of arguments to the intialiser
 *
 * Returns a copy of the PyObject it instantiates.
 */
static PyObject *rendermodule(PyObject *self, PyObject *args) {
  PyArrayObject *x_obj, *y_obj, *t_obj;
  PyArrayObject *m_obj;
  float *x, *y, *t;
  float *mass;
  int xsize, ysize;
  int n;
  float *image;
  int DOUBLE = 0;

  if (!PyArg_ParseTuple(args, "OOOOii", &x_obj, &y_obj, &t_obj, &m_obj, &xsize,
                        &ysize))
    return NULL;
    
  // Let's check the size of the 1-dimensions arrays.
  n = (int) m_obj->dimensions[0];

  // Let's point the C arrays to the numpy arrays
  x = (float *)x_obj->data;
  y = (float *)y_obj->data;
  t = (float *)
          t_obj->data; /* These are always floats, as they come from Scene */

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

  image = (float *)malloc( xsize * ysize * sizeof(float) );

  int i, j;
  for(i=0;i<xsize;i++){
    for(j=0;j<ysize;j++){
      image[j*xsize+i] = 0.0;
    }
  }

  // Here we do the work
  c_render(x,y,t,mass,xsize,ysize,n,image);

  if(DOUBLE) free(mass);
  
  // Let's build a numpy array
  npy_intp dims[1] = {xsize*ysize};
  PyArrayObject *image_obj = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, image);
  image_obj->flags = NPY_OWNDATA;
  
  return Py_BuildValue("N", image_obj);
		       
}

/*! Global variable that registers the rendermodule method internally
    in python */
static PyMethodDef RenderMethods[] = {
    {"render", rendermodule, METH_VARARGS, "Method for rendering the image."},
    {NULL, NULL, 0, NULL}};

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

/* Set me to 128 to run C debugging */
#define C_DEBUG_MODE_NO_PYTHON 0
#if C_DEBUG_MODE_NO_PYTHON == 128
/**
 * @brief Function for testing, not used in production code
 */
void test_C() {
  // This function if for testing purposes only. It writes a file called
  // image_test.bin
  int *x, *y, *t;
  float *mass;
  int xsize, ysize, n;
  float *image;
  int i;

  xsize = 1000;
  ysize = 1000;
  n = 10000;

  x = (int *)malloc(n * sizeof(int));
  y = (int *)malloc(n * sizeof(int));
  t = (int *)malloc(n * sizeof(int));
  mass = (float *)malloc(n * sizeof(float));
  image = (float *)malloc(xsize * ysize * sizeof(float));

  srand(time(NULL));

  for (i = 0; i < n; i++) {
    x[i] = rand() % xsize;
    y[i] = rand() % ysize;
    t[i] = rand() % 50;
    mass[i] = rand() % 499;
  }

  c_render(x, y, t, mass, xsize, ysize, n, image);

  FILE *output;

  output = fopen("image_test.bin", "wb");
  fwrite(image, sizeof(float), xsize * ysize, output);
  fclose(output);
}

/* Main function for testing in C-only mode */
void main() { test_C(); }
#endif /*C_DEBUG_MODE_NO_PYTHON*/
