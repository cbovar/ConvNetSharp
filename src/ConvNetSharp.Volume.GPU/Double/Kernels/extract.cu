extern "C" {
    __global__ void Run(int n, double* __restrict x, double* __restrict output, int length, int offset, int inputCount) {
		int i = blockIdx.x*blockDim.x + threadIdx.x;
        
        int batch = i / length;
        int rest = i % length;

        int batchCount = n / length;

        int inputWidth = inputCount / batchCount;

		if (i < n) {
            output[i] = x[batch * inputWidth + rest + offset];
        }
	}
}