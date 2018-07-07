extern "C" {
    __global__ void Run(int n, float* __restrict left, float* __restrict right, float* __restrict output, int rightIsScalar) {
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		if (i < n) {
            if (rightIsScalar == 1) { 
                output[i] = left[i] / right[0];
            } else {
                output[i] = left[i] / right[i];
            }
        }
	}
}