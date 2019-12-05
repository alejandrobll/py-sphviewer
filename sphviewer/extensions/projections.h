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

long int ortographic_projection(float *x, float *y, float *z, float *hsml, long int *kview,
				int n, float *extent, int xsize, int ysize){
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
    hsml[i] = hsml[i]/(xmax-xmin) * xsize;
  }
  
  return idx;
}

long int oblique_projection(float *x, float *y, float *z, float *hsml, long int *kview,
			    int n, float r, float zoom, float *extent, int xsize, int ysize){

  long int idx;
  
  float FOV = 2.0*fabsf(atanf(1.0/zoom));
  float xmax = 1.0;
  float xmin = -xmax;
  float ymax = 0.5*(xmax-xmin)*ysize/xsize;   /* Let's preserve the pixel aspect ration */
  float ymin = -ymax;
  
  float xfovmax = FOV/2.*180.0/3.141592;
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
    hsml[i] = (hsml[i]/zpart)/(xmax-xmin) * xsize;
  }
  
  return idx;
}

    

