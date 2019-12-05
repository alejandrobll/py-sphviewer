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

#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define M_PI acos(-1.0)

long int ortographic_projection(float *x, float *y, float *z,
				float *hsml, float *hsml_x, float *hsml_y,
				long int *kview, int n, float *extent,
				int xsize, int ysize){
  float xmin = extent[0];
  float xmax = extent[1];
  float ymin = extent[2];
  float ymax = extent[3];

  long int idx = 0;
  int i;

  for(i=0 ;i<n;i++){
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
    hsml_x[i] = hsml_x[i]/(xmax-xmin) * xsize;
    hsml_y[i] = hsml_y[i]/(ymax-ymin) * ysize;
  }
  
  return idx;
}

long int oblique_projection(float *x, float *y, float *z,
			    float *hsml, float *hsml_x, float *hsml_y,
			    long int *kview, int n, float r, float zoom,
			    float *extent, int xsize, int ysize){

  long int idx;
  
  float FOV = 2.0*fabsf(atanf(1.0/zoom));
  float xmax = 1.0;
  float xmin = -xmax;
  float ymax = 0.5*(xmax-xmin)*ysize/xsize;   /* Let's preserve the pixel aspect ration */
  float ymin = -ymax;
  
  float xfovmax = FOV/2.*180.0/M_PI;
  float xfovmin = -xfovmax;
  float yfovmax = 0.5*(xfovmax-xfovmin)*ysize/xsize;
  float yfovmin = -yfovmax;
  
  extent[0] = xfovmin;
  extent[1] = xfovmax;
  extent[2] = yfovmin;
  extent[3] = yfovmax;
  
  float zpart;
  int i;
  for(i=0;i<n;i++){
    zpart = (z[i]-(-1.0*r));
    if( (zpart > 0) & 
	(fabsf(x[i]) <= fabsf(zpart)*xmax/zoom) &
	(fabsf(y[i]) <= fabsf(zpart)*ymax/zoom) ) {
      
      kview[idx] = i;
      idx += 1;
    }
  }
  
#pragma omp parallel for firstprivate(n,xmin,xmax,ymin,ymax,xsize,ysize,zoom,r,zpart)
  for(i=0;i<n;i++){
    zpart = (z[i]-(-1.0*r))/zoom;
    x[i] = (x[i]/zpart - xmin) / (xmax-xmin) * xsize;
    y[i] = (y[i]/zpart - ymin) / (ymax-ymin) * ysize;
    hsml_x[i] = (hsml_x[i]/zpart)/(xmax-xmin) * xsize;
    hsml_y[i] = (hsml_y[i]/zpart)/(ymax-ymin) * ysize;
  }
  
  return idx;
}

void project_onto_sphere(float *x, float *y, float *z,
			 float *hsml, int n){
  /* this function projects the particles onto a sphere of radius 1, 
     and replaces x and y with the spherical coordinates phi 
     and theta, respectively */
  
  int i;
  float r, theta, phi;
  
#pragma omp parallel for firstprivate(n,r,theta, phi)
  for(i=0; i<n; i++){
    r = sqrtf(x[i]*x[i]+y[i]*y[i]+z[i]*z[i]);
    theta = acosf(z[i]/r);
    phi =  atan2f(y[i], x[i]);
    x[i] = phi;
    y[i] = theta;
    hsml[i] = 2.0 * atanf(0.5 * hsml[i]/r);
  }
  return;
}


long int equirectangular_projection(float *x, float *y, float *z,
				    float *hsml, float *hsml_x, float *hsml_y,
				    long int *kview, int n, float r, float zoom,
				    float *extent, int xsize, int ysize){

  /*First, project particles onto the sphere */
  project_onto_sphere(x, y, z, hsml, n);
  
  /* Now calculate the actual projection */
  
  int i;
  
  float xmin = 0.0;
  float xmax = 2*M_PI;
  float ymax = M_PI;  
  float ymin = 0;
    
  extent[0] = xmin;
  extent[1] = xmax;
  extent[2] = ymin;
  extent[3] = ymax;
    
#pragma omp parallel for firstprivate(n,xmin,xmax,ymin,ymax,xsize,ysize)
  for(i=0;i<n;i++){
    x[i] = (x[i] + M_PI) / (2.0 * M_PI) * xsize;
    y[i] = y[i] / M_PI *  ysize;
    hsml_x[i] = hsml[i] / (2.0 * M_PI) * xsize; /* kernel x */
    hsml_y[i] = hsml[i] / M_PI * ysize; /* kernel y */
    hsml[i]   = hsml[i] / (2.0 * M_PI) * xsize; /* angular distance kernel */
    kview[i] = i;
  }  
  return (long int) n;
}



