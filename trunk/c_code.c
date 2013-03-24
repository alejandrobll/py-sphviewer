int i, j, k;
int xx, yy;
int ttx, tty; 
float rho_c;
int binx_c, biny_c;

binx_c = binx;
biny_c = biny;

#pragma omp parallel for private(i,xx,yy,ttx,tty,rho_c,j,k)
#pragma omp+ reduction(+:dens)
#pragma omp schedule(dynamic,1000)
	for(i=0;i<n;i++)
	{
		//		  printf("Queda %d\\n",n-i);
		xx = (int)x(i);
		yy = (int)y(i);
		ttx = (int)tx(i);
		tty = (int)ty(i);
		rho_c = (float) rho(i);

		if(ttx < 1) ttx = 1;
		if(ttx > binx_c) ttx = binx_c;

		if(tty < 1) tty = 1;
		if(tty > biny_c) tty = biny_c;

		for(j=-ttx; j<ttx+1; j++)
		{
			for(k=-tty; k<tty+1; k++)
			{
				if( ( (xx+j) >= 0) && ( (xx+j) < binx_c) && ( (yy+k) >=0) && ( (yy+k) < biny_c))
				dens((xx+j),(yy+k)) += rho_c*cubic_kernel2(sqrt((float)j*(float)j+(float)k*(float)k), 1.0/sqrt(2.0)*sqrt( (float)tty*(float)tty+(float)ttx*(float)ttx));
			}
		}
	}
