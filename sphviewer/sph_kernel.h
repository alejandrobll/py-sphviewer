/* This is where we define the SPH kernel. This is a little
 * naughty, as we're not packging stuff in a C file, but rather
 * a .h. Oh well. */

/* Standard includes */
#include <math.h>

#ifndef SPHVIEWER_SPH_KERNEL_H
#define SPHVIEWER_SPH_KERNEL_H

/* Choose now, or forever hold your peace:
 * 
 * 1. Cubic spline (Monaghan & Lattanzio 1985)
 * 2. Cubic spline (Monaghan & Lattanzio 1985) in a smaller domain
 * 3. 2D dome-shaped quadratic kernel (old default)
 * 4.
 */
#define SPH_VISUALISATION_KERNEL_CHOICE 3

#if SPH_VISUALISATION_KERNEL_CHOICE == 1

/**
 * @brief: 3D Piecewise cubic spline (Monaghan and Lattanzio, 1985).
 * 
 * @param r: Distance between particle and cell center
 * @param h: Smoothing length of the particle.
 */
float cubic_kernel(float r, float h)
{
  const float h_inv = 1.f / h;
  const float h_inv_3 = h_inv * h_inv * h_inv;
  const float ratio = r * h_inv;
  const float sigma = 1.f / 3.141592f;
  float func = 0.f;

  if (ratio < 1.f)
  {
    const float ratio_2 = ratio * ratio;
    const float ratio_3 = ratio_2 * ratio;
    func = 1.f - 1.5f * ratio_2 + 0.75f * ratio_3;
  }
  else if (ratio < 2.f)
  {
    const float two_minus_ratio = 2.f - ratio;
    func = 0.25f * two_minus_ratio * two_minus_ratio * two_minus_ratio;
  }

  return sigma * h_inv_3 * func;
}

#elif SPH_VISUALISATION_KERNEL_CHOICE == 2

/**
 * @brief: 3D Piecewise cubic spline (Monaghan and Lattanzio, 1985)
 *         in a smaller domain.
 * 
 * @param r: Distance between particle and cell center
 * @param h: Smoothing length of the particle.
 */

float cubic_kernel(float r, float h)
{
  const float h_inv = 1.f / h;
  const float h_inv_3 = h_inv * h_inv * h_inv;
  const float ratio = r * h_inv;
  const float sigma = 8.f / 3.141592f;
  float func = 0.f;

  if (ratio < 0.5f)
  {
    const float ratio_2 = ratio * ratio;
    const float ratio_3 = ratio_2 * ratio;
    func = 1.f - 6.f * ratio_2 + 6.f * ratio_3;
  }
  else if (ratio < 1.f)
  {
    const float one_minus_ratio = 1.f - ratio;
    func = 2.f * one_minus_ratio * one_minus_ratio * one_minus_ratio;
  }

  return sigma * h_inv_3 * func;
}

#elif SPH_VISUALISATION_KERNEL_CHOICE == 3

/**
 * @brief: 2D dome-shaped quadratic kernel (1 - R^2) for visualisation
 *         (see Hicks & Liebrock 2000).
 *
 * @param r: Distance between particle and cell center
 * @param h: Smoothing length of the particle.
 */
float cubic_kernel(float r, float h)
{
  const float h_inv = 1.f / h;
  const float h_inv_2 = h_inv * h_inv;
  const float ratio = r * h_inv;
  const float sigma = 15.0 / (8.0 * 3.141592);

  float func = 0.f;

  if (ratio <= 1.f)
  {
    const float argument = 1.f - ratio * ratio;
    /* Trying to use an inverse square root trick here doesn't work.
     * No idea why. Perhaps worth checking later,
     * argument_3_2 = argument * argument / sqrtf(argument)
     * (this didn't work for gcc9.1.0 or gcc7.x) */
    const float argument_3_2 = argument * sqrtf(argument);
    func = (4.f / 3.f) * h_inv_2 * argument_3_2;
  }

  return func;
}

#elif SPH_VISUALISATION_KERNEL_CHOICE == 4

/**
 * @brief: 2D dome-shaped quadratic kernel (1 - R^2) for visualisation
 *         (see Hicks & Liebrock 2000). This one is simpler than the above
 *         but 'should only be used for fancy plots'
 *
 * @param r: Distance between particle and cell center
 * @param h: Smoothing length of the particle.
 */
float cubic_kernel(float r, float h)
{
  float func = 0.f;
  float ratio = r / h;

  if (ratio < 1.f)
  {
    func = 1.f - ratio * ratio;
  }

#else
#error "Invalid choice of SPH kernel, see sph_kernel.h"
#endif /* SPH_VISUALISATION_KERNEL_CHOICE */

#endif /* SPHVIEWER_SPH_KERNEL_H */