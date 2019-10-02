/* Main rendering loops. This is where the SPH-magic happens! */

#include "sph_kernel.h"

const int xsize_c, ysize_c = {xsize, ysize};
/* What's the largest dimension? */
const float size_lim = (float)(xsize_c > ysize_c ? xsize_c : ysize_c);

/* Parallel loop; each thread has its own thread-local image that are combined
 * at the end. Note that everything here is done in terms of the pixel grid
 * co-ordinates. */
#pragma omp parallel
{
  const int n_threads = omp_get_num_threads();
  const int thread_id = omp_get_thread_num();
  /* Particles per thread, n is total number of particles. */
  const int particles_per_thread =
      n / n_threads; // Number or Particles per thread (ppt)
  const int remainder_particles = n - particles_per_thread * n_threads;

  /* Allocate the local image that we'll spread our particles onto and zero it
   */
  float *local_image = (float *)malloc(xsize_c * ysize_c * sizeof(float));
  bzero(local_image, xsize_c * ysize_c * sizeof(float));

  /* Loop over the particles only belonging to our thread */
  for (int i = (thread_id * particles_per_thread);
       i < (thread_id + 1) * particles_per_thread; i++) {

    const float x_float = x(i);
    const float y_float = y(i);
    const int x_cell = (int)x_float;
    const int y_cell = (int)y_float;
    float smoothing_length = t(i);
    float particle_mass = (float)mass(i);

    if (smoothing_length < 0.5f) {
      /* If the smoothing length is less than half of a pixels width, we
       * can contribute in a very simplistic way to the density of the pixel
       * which we know has area A = 1. */

      const int pixel = cell_y * xsize_c + cell_x;
      local_image[pixel] += particle_mass; /* I.e. mass / A with A=1 */

      /* We're done! */
      continue;
    }

    /* Curtail the smoothing length if it is to large */
    if (smoothing_length > size_lim) {
      smoothing_length = size_lim;
    }

    /* Cast the smoothing length to an integer so we can loop over the pixel
     * square cast by it */
    const int pixels_to_loop_over = (int)smoothing_length;

    /* Loop over the pixels that our kernel covers? */
    for (int j = -pixels_to_loop_over; j < pixels_to_loop_over + 1; j++) {
      /* Need to check if we live within the bounds of the image */
      if ((x_cell + j >= 0) && (x_cell + j < xsize_c)) {
        /* Compute x-only properties that will stay constant for y loop */
        const float distance_x = ((float)x_cell + (float)j - 0.5f) - x_float;
        const float distance_x_2 = distance_x * distance_x;

        for (int k = -pixels_to_loop_over; k < pixels_to_loop_over + 1; k++) {
          if ((y_cell + k >= 0) && (y_cell + k < ysize_c)) {
            const float distance_y =
                ((float)y_cell + (float)k - 0.5f) - y_float;
            const float distance_y_2 = distance_y * distance_y;

            const float radius = sqrtf(distance_y_2 + distance_x_2);
            /* Can call the kernel! Woo! */
            const float kernel = cubic_kernel(radius, smoothing_length);

            /* Now add onto the correct cell */
            const int pixel = (cell_y + k) * xsize_c + (cell_x + j);
            local_image[pixel] += particle_mass * kernel;
          }
        }
      }
    }
  }

  // Let's compute the image for the remainder particles...
  if ((r - thread_id) > 0) {
    const int i = n_threads * particles_per_thread + thread_id;
    const float x_float = x(i);
    const float y_float = y(i);
    const int x_cell = (int)x_float;
    const int y_cell = (int)y_float;
    float smoothing_length = t(i);
    float particle_mass = (float)mass(i);

    if (smoothing_length < 0.5f) {
      /* If the smoothing length is less than half of a pixels width, we
       * can contribute in a very simplistic way to the density of the pixel
       * which we know has area A = 1. */

      const int pixel = cell_y * xsize_c + cell_x;
      local_image[pixel] += particle_mass; /* I.e. mass / A with A=1 */

      /* We're done! */
    } else {
      /* Gotta do the smoothing as usual */

      if (smoothing_length > size_lim) {
        smoothing_length = size_lim;
      }

      /* Cast the smoothing length to an integer so we can loop over the pixel
       * square cast by it */
      const int pixels_to_loop_over = (int)smoothing_length;

      /* Loop over the pixels that our kernel covers? */
      for (int j = -pixels_to_loop_over; j < pixels_to_loop_over + 1; j++) {
        /* Need to check if we live within the bounds of the image */
        if ((x_cell + j >= 0) && (x_cell + j < xsize_c)) {
          /* Compute x-only properties that will stay constant for y loop */
          const float distance_x = ((float)x_cell + (float)j - 0.5f) - x_float;
          const float distance_x_2 = distance_x * distance_x;

          for (int k = -pixels_to_loop_over; k < pixels_to_loop_over + 1; k++) {
            if ((y_cell + k >= 0) && (y_cell + k < ysize_c)) {
              const float distance_y =
                  ((float)y_cell + (float)k - 0.5f) - y_float;
              const float distance_y_2 = distance_y * distance_y;

              const float radius = sqrtf(distance_y_2 + distance_x_2);
              /* Can call the kernel! Woo! */
              const float kernel = cubic_kernel(radius, smoothing_length);

              /* Now add onto the correct cell */
              const int pixel = (cell_y + k) * xsize_c + (cell_x + j);
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
    for (j = 0; j < xsize_c; j++) {
      for (k = 0; k < ysize_c; k++) {
        image(k, j) += local_image[k * xsize_c + j];
      }
    }
    free(local_image);
  }
}
