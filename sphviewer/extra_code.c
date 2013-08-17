// file with different kernels

float cubic_kernel1(float r, float h)
//3D Piecewise cubic spline (Monaghan and Lattanzio, 1985).
{
  float func;
  float sigma;
  
  sigma = 1.0/3.141592;
  if((r/h) < 1.0)  func = 1.0-1.5*(r/h)*(r/h)+0.75*(r/h)*(r/h)*(r/h);
  if( ((r/h) < 2.0) && ((r/h) >= 1.0) )  func = 0.25*(2.0-(r/h))*(2.0-(r/h))*(2.0-(r/h));
  if((r/h) >= 2.0) func = 0.0;
  return sigma/(h*h*h)*func;
}
float cubic_kernel2(float r, float h)
//3D Piecewise cubic spline (Monaghan and Lattanzio, 1985) in a 
//smaller domain.
{
  float func;
  float sigma;
  float h2 = h*h;
  float h3 = h*h*h;
  float R  = r/h;

  sigma = 8.0/3.141592;
    
  if(R <= 0.5) func = 1.0-6.0*(R*R)+6.0*(R*R*R);
  if( (R > 0.5) && (R <= 1.0)) func = 2.*(1.0-R)*(1.0-R)*(1.0-R);
  if(R > 1.0)	func = 0.0;

  return sigma/(h3)*func;
}

float cubic_kernel3(float r, float h)
//2D Dome-shapeed quadratic Kernel (1-R^2) (Hicks and Liebrock 2000).
{
  float func;
  float sigma;

  sigma = 15.0/(8.0*3.141592);
  if(r/h <= 1.0)  
    func = 4.0/3.0*h*pow(sqrt(1.0-(r/h)*(r/h)),3);
  if(r/h > 1.0)  
    func = 0;
  return sigma*func/(h*h*h);
}

float cubic_kernel4(float r, float h)
//3D Dome-shapeed quadratic Kernel (1-R^2) (Hicks and Liebrock 2000).
//It should be used only for fancy plots!!
{
  float func;
  if(r/h < 1.0)  
    func = 1.0-(r/h)*(r/h);
  if(r/h > 1.0)  
    func = 0;
  return func;
}

void hsvToRgb (float h, float s, float v, float * r, float * g, float * b)
{
  int i;
  float aa, bb, cc, f;

  if (s == 0) /* Grayscale */
    *r = *g = *b = v;
  else {
    if (h == 1.0) h = 0;
    h *= 6.0;
    i = floor(h); 
    f = h - i;
    aa = v * (1 - s);
    bb = v * (1 - (s * f));
    cc = v * (1 - (s * (1 - f)));
    switch (i) {
    case 0: *r = v;  *g = cc; *b = aa; break;
    case 1: *r = bb; *g = v;  *b = aa; break;
    case 2: *r = aa; *g = v;  *b = cc; break;
    case 3: *r = aa; *g = bb; *b = v;  break;
    case 4: *r = cc; *g = aa; *b = v;  break;
    case 5: *r = v;  *g = aa; *b = bb; break;
    }
  }
}


