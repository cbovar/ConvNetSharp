extern "C" {
    __global__ void Run(int n, float* __restrict left, float* __restrict right, float* __restrict output, int elementPerBatch, int threshold, int mode) {
		int i = blockIdx.x*blockDim.x + threadIdx.x;
        int batch = i / elementPerBatch;
        int rest = i % elementPerBatch;
		if (i < n) {
            if (rest < threshold) {
                if (mode == 1) {
                    output[i] = left[0];
                } else {
                    output[i] = left[batch * threshold + rest];
                }
            } else {
                if (mode == 2) {
                    output[i] = right[0];
                } else {
                    output[i] = right[batch * (elementPerBatch - threshold) + rest - threshold];
                }
            }
        }
	}
}