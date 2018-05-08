extern "C" {
    __global__ void Run(int n, float* __restrict input, float* __restrict output, float alpha) {
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		if (i < n) output[i] =  input[i] > 0 ? input[i] : input[i] * alpha; 
	}
}