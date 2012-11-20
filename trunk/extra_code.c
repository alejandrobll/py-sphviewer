// file with different kernels

float cubic_kernel1(float r, float h)
{
	float func;
	if((r/h) < 1.0)  func = 1.0-1.5*(r/h)*(r/h)+0.75*(r/h)*(r/h)*(r/h);
	if( ((r/h) < 2.0) && ((r/h) >= 1.0) )  func = 0.25*(2.0-(r/h))*(2.0-(r/h))*(2.0-(r/h));
	if((r/h) >= 2.0) func = 0.0;
	return func;
}
float cubic_kernel2(float r, float h)
{
	float func;
	if((r/h) <= 0.5) func = 1.0-6.0*(r/h)*(r/h)+6.0*(r/h)*(r/h)*(r/h);
	if( ((r/h) > 0.5) && (r/h) <= 1.0) func = 2.*(1.0-(r/h))*(1.0-(r/h))*(1.0-(r/h));
	if((r/h) > 1.0)	func = 0.0;
	return func;
}

