extern "C" {
    __global__ void Run(int n, float* __restrict left, float* __restrict right, float* __restrict output) {
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		if (i < n) output[i] = left[i] / right[i];
	}
}