extern "C" {
    __global__ void Run(int n, float* __restrict x, float* __restrict output, int length, int offset) {
		int i = blockIdx.x*blockDim.x + threadIdx.x;
        
        int batch = i / length;
        int rest = i % length;
        int batchCount = n / length;
        int width = sizeof(x) / batchCount;

		if (i < n) {
            output[i] = x[batch * width + rest + offset];
        }
	}
}