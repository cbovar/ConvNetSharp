extern "C" {
    __global__ void Run(int pad, int stride, double* __restrict biasesWeights,
						int filterWidth, int filterHeight, int filterDepth, double* __restrict filterWeights,
						int inputWidth, int inputHeight, int inputDepth, double* __restrict inputWeigths,
						int outputWidth, int outputHeight, int outputDepth, double* __restrict outputWeights) {

            int t = blockIdx.x * blockDim.x + threadIdx.x;
            int total = outputWidth * outputHeight;
            int temp = t % total;
            int depth = t / total;
            int ax = temp % outputWidth;
            int ay = temp / outputWidth;

            if (depth < outputDepth && ax < outputWidth && ay < outputHeight)
            {
                int totalfilter = filterHeight * filterWidth * filterDepth;
                int x = -pad + ax * stride;
                int y = -pad + ay * stride;

                // convolve centered at this particular location
                double a = 0.0;
                for (int fy = 0; fy < filterHeight; fy++)
                {
                    int oy = y + fy; // coordinates in the original input array coordinates
                    for (int fx = 0; fx < filterWidth; fx++)
                    {
                        int ox = x + fx;
                        if (oy >= 0 && oy < inputHeight && ox >= 0 && ox < inputWidth)
                        {
                            for (int fd = 0; fd < filterDepth; fd++)
                            {
                                a += filterWeights[depth * totalfilter + ((filterWidth * fy) + fx) * filterDepth + fd] *
                                     inputWeigths[((inputWidth * oy) + ox) * inputDepth + fd];
                            }
                        }
                    }
                }

                a += biasesWeights[depth];
                int ix = ((outputWidth * ay) + ax) * outputDepth + depth;
                outputWeights[ix] = a;
            }
        }
}