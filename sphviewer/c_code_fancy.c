int i, j, k, l;
int xx, yy;
int tt; 
float mass_c;
int xsize_c, ysize_c;
int size_lim;
float prop1_c, prop2_c;
float r,g,b;
float value;

xsize_c = xsize;
ysize_c = ysize;

if(xsize_c >= ysize_c) size_lim = xsize_c;
if(xsize_c < ysize_c)  size_lim = ysize_c;

#pragma omp parallel for private(i,xx,yy,tt,mass_c,j,k,prop1_c,prop2_c,value,r,g,b)
#pragma omp+ reduction(+:image)
#pragma omp schedule(dynamic,1000)
for(i=0;i<n;i++){
  xx = (int)x(i);
  yy = (int)y(i);
  tt = (int)t(i);
  mass_c = (float) mass(i);
  prop1_c = (float) prop1(i);
  prop2_c = (float) prop2(i);

  if(tt < 1) tt = 1;
  if(tt > size_lim) tt = size_lim;

  for(j=-tt; j<tt+1; j++){
    for(k=-tt; k<tt+1; k++){
      if( ( (xx+j) >= 0) && ( (xx+j) < xsize_c) && ( (yy+k) >=0) && ( (yy+k) < ysize_c)){
	value =  mass_c*cubic_kernel4(sqrt((float)j*(float)j+(float)k*(float)k), tt);
	hsvToRgb(prop1_c, prop2_c, value, &r, &g, &b);
	image(yy+k,xx+j,0) = 1.0-(1.0-image(yy+k,xx+j,0))*(1.0-r);
	image(yy+k,xx+j,1) = 1.0-(1.0-image(yy+k,xx+j,1))*(1.0-g);
	image(yy+k,xx+j,2) = 1.0-(1.0-image(yy+k,xx+j,2))*(1.0-b);
      }
    }
  }
 }

