int i, j, k;
int xx, yy;
int tt; 
float rho_c;
int binx_c, biny_c;
int bin_lim;

binx_c = binx;
biny_c = biny;

if(binx_c >= biny_c) bin_lim = binx_c;
if(binx_c < biny_c) bin_lim = biny_c;

#pragma omp parallel for private(i,xx,yy,tt,rho_c,j,k)
#pragma omp+ reduction(+:dens)
#pragma omp schedule(dynamic,1000)
	for(i=0;i<n;i++)
	{
		//		  printf("Queda %d\\n",n-i);
		xx = (int)x(i);
		yy = (int)y(i);
		tt = (int)t(i);
		rho_c = (float) rho(i);

		if(tt < 1) tt = 1;
		if(tt > bin_lim) tt = bin_lim;

		for(j=-tt; j<tt+1; j++)
		{
			for(k=-tt; k<tt+1; k++)
			{
				if( ( (xx+j) >= 0) && ( (xx+j) < binx_c) && ( (yy+k) >=0) && ( (yy+k) < biny_c))
				dens((yy+k),(xx+j)) += rho_c*cubic_kernel2(sqrt((float)j*(float)j+(float)k*(float)k), tt);
			}
		}
	}
