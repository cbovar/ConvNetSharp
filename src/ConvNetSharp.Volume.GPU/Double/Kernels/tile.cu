extern "C" {
     __global__ void Run(int n, double* __restrict input, double* __restrict output, 
                         int inputWidth, int inputHeight, int inputChannel, int inputBatchsize,
                         int outputWidth, int outputHeight, int outputChannel, int outputBatchsize) {
		int i = blockIdx.x*blockDim.x + threadIdx.x;
        if (i < n) {
            // compute input coordinates 
            int channelTotal = inputWidth * inputHeight;
            int batchTotal = channelTotal * inputChannel;

            int b = i / batchTotal;

            int c = (i % batchTotal ) / channelTotal;
            int y = ((i % batchTotal) % channelTotal) / inputWidth;
            int x = ((i % batchTotal) % channelTotal) % inputWidth;

            // use modulus to create tiles
            int inputX = x % inputWidth;
            int inputY = y % inputHeight;
            int inputC = c % inputChannel;
            int inputB = b % inputBatchsize;
            
            output[i] = input[inputX + inputY * inputWidth + inputC * channelTotal + inputB * batchTotal];
        }
	}
}