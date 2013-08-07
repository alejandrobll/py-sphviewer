int i, j, k, l;
int xx, yy;
int tt; 
float mass_c;
int xsize_c, ysize_c;
int size_lim;

xsize_c = xsize;
ysize_c = ysize;

if(xsize_c >= ysize_c) size_lim = xsize_c;
if(xsize_c < ysize_c)  size_lim = ysize_c;

#pragma omp parallel for private(i,xx,yy,tt,mass_c,j,k)
#pragma omp+ reduction(+:image)
#pragma omp schedule(dynamic,1000)
for(i=0;i<n;i++){
  xx = (int)x(i);
  yy = (int)y(i);
  tt = (int)t(i);
  mass_c = (float) mass(i);

  if(tt < 1) tt = 1;
  if(tt > size_lim) tt = size_lim;

//  if(tt == 1){
//    image(yy,xx) = mass_c;
//  }
//  if(tt > 1){
    for(j=-tt; j<tt+1; j++){
      for(k=-tt; k<tt+1; k++){
	if( ( (xx+j) >= 0) && ( (xx+j) < xsize_c) && ( (yy+k) >=0) && ( (yy+k) < ysize_c)){
	  image(yy+k,xx+j) += mass_c*cubic_kernel3(sqrt((float)j*(float)j+(float)k*(float)k), tt);
	}
      }
//    }
  }
 }

