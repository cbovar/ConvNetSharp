extern "C" {
    __global__ void Run(int n, double* __restrict input, double* __restrict gradient, double* __restrict output) {
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		if (i < n) output[i] =  gradient[i] * (input[i] >= 0) + gradient[i] * (input[i] < 0) * 0.01;
	}
}