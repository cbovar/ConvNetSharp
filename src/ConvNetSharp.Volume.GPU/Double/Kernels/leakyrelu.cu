extern "C" {
    __global__ void Run(int n, double* __restrict input, double* __restrict output) {
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		if (i < n) output[i] =  input[i] > 0 ? input[i] : input[i] * 0.01; 
	}
}