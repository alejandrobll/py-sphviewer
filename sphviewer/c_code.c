int xsize_c, ysize_c;
int size_lim;

xsize_c = xsize;
ysize_c = ysize;

if(xsize_c >= ysize_c) size_lim = xsize_c;
if(xsize_c < ysize_c)  size_lim = ysize_c;

#pragma omp parallel
{
  float* local_image;
  int i, j, k, l;
  int xx, yy;
  int tt; 
  float mass_c;
  int r, nth, ppt, thread_id;

  nth = omp_get_num_threads(); 
  thread_id = omp_get_thread_num();
  ppt = n/nth;      // Number or Particles per thread (ppt)
  r = n-ppt*nth;    // Remainder
  /*  if(thread_id == 0){
    printf("Numero de threads = %d\n", nth);
    printf("Numero de particulas por thread = %d\t%d\n", ppt,r);
    }*/
  // Let's compute the image
  local_image = (float*)malloc(xsize_c*ysize_c*sizeof(float));

  for(j=0;j<xsize_c;j++){
    for(k=0;k<ysize_c;k++){
      local_image[k*xsize_c+j] = 0.0;
    }
  }

    for(i=(thread_id*ppt); i<(thread_id+1)*ppt; i++){
      xx = (int)x(i);
      yy = (int)y(i);
      tt = (int)t(i);
      mass_c = (float) mass(i);

      if(tt < 1) tt = 1;
      if(tt > size_lim) tt = size_lim;

      for(j=-tt; j<tt+1; j++){
	for(k=-tt; k<tt+1; k++){
	  if( ( (xx+j) >= 0) && ( (xx+j) < xsize_c) && ( (yy+k) >=0) && ( (yy+k) < ysize_c)){
	    local_image[(yy+k)*xsize_c+(xx+j)] += mass_c*cubic_kernel3(sqrt((float)j*(float)j+(float)k*(float)k), tt);
	  }
	}
      }
    }
   
  // Let's compute the image for the remainder particles...
  if((r-thread_id) > 0){
    i  = nth*ppt+thread_id;
    xx = (int)x(i);
    yy = (int)y(i);
    tt = (int)t(i);
    mass_c = (float) mass(i);

    if(tt < 1) tt = 1;
    if(tt > size_lim) tt = size_lim;

    for(j=-tt; j<tt+1; j++){
      for(k=-tt; k<tt+1; k++){
	if( ( (xx+j) >= 0) && ( (xx+j) < xsize_c) && ( (yy+k) >=0) && ( (yy+k) < ysize_c)){
	  local_image[(yy+k)*xsize_c+(xx+j)] += mass_c*cubic_kernel3(sqrt((float)j*(float)j+(float)k*(float)k), tt);
	}
      }
    }
  }
  #pragma omp critical
  {
    for(j=0;j<xsize_c;j++){
      for(k=0;k<ysize_c;k++){
	image(k,j) += local_image[k*xsize_c+j];
      }
    }
    free(local_image);
  }
}
