
// GPU kernel
__global__ void process(float * data_out, int gap)
{
	float res = 0.;
	int numthread = blockIdx.x * blockDim.x + threadIdx.x;
	bool pair = (((numthread*gap + gap-1) % 2) ==0);
	for(int i = (numthread*gap+gap-1); i >= (numthread*gap); i--){
		res += (pair?1.:-1.)/(i+1.);
		pair = !pair;
	}
	data_out[numthread] = res;
}


