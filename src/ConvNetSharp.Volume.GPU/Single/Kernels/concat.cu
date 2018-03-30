extern "C" {
    __global__ void Run(int n, float* __restrict left, float* __restrict right, float* __restrict output, int elementPerBatch, int threshold) {
		int i = blockIdx.x*blockDim.x + threadIdx.x;
        int batch = i / elementPerBatch;
        int rest = i % elementPerBatch;
		if (i < n) {
            if (rest < threshold) {
                output[i] = left[batch * threshold + rest];
            } else {
                output[i] = right[batch * (elementPerBatch - threshold) + rest - threshold];
            }
        }
	}
}