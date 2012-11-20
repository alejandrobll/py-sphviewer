int i, j, k;
int xx, yy;
int tt; 
float rho_c;
int binx_c, biny_c;

binx_c = binx;
biny_c = biny;

#pragma omp parallel for private(i,xx,yy,tt,rho_c,j,k)
#pragma omp+ reduction(+:dens)
#pragma omp schedule(dynamic)
	for(i=0;i<n;i++)
	{
		//		  printf("Queda %d\\n",n-i);
		xx = (int)x(i);
		yy = (int)y(i);
		tt = (int)t(i);
		rho_c = (float) rho(i);

		if(tt < 1) tt = 1;
		if(tt > binx_c) tt = binx_c;
		for(j=-tt; j<tt+1; j++)
		{
			for(k=-tt; k<tt+1; k++)
			{
				if( ( (xx+j) >= 0) && ( (xx+j) < binx_c) && ( (yy+k) >=0) && ( (yy+k) < biny_c))
				dens((xx+j),(yy+k)) += rho_c*cubic_kernel2(sqrt((float)j*(float)j+(float)k*(float)k), tt);
			}
		}
	}
