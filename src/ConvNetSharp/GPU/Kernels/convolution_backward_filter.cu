extern "C" {

 __global__ void Run(int pad, int stride,
            int filterWidth, int filterHeight, int filterDepth, double* __restrict filterGradients,
            int inputWidth, int inputHeight, int inputDepth, double* __restrict inputWeigths,
            int outputWidth, int outputHeight, int outputDepth, double* __restrict outputGradients)
        {
            int t = blockIdx.x * blockDim.x + threadIdx.x;

            int total1 = filterWidth * filterWidth * filterDepth;
            int temp1 = t % total1;
            int depth = t / total1;

            int total = filterWidth * filterWidth;
            int temp = temp1 % total;
            int fd = temp1 / total;
            int fy = temp / filterWidth;
            int fx = temp % filterWidth;

            if (fd < filterDepth && depth < outputDepth)
            {
                int totalfilter = filterHeight * filterWidth * filterDepth;

                double a = 0;
                for (int ay = 0; ay < outputHeight; ay++)
                {
                    for (int ax = 0; ax < outputWidth; ax++)
                    {
                        int ix = ((outputWidth * ay) + ax) * outputDepth + depth;
                        double chainGradient = outputGradients[ix];

                        int x = -pad + ax * stride;
                        int y = -pad + ay * stride;

                        // convolve centered at this particular location
                        int oy = y + fy; // coordinates in the original input array coordinates
                        int ox = x + fx;
                        if (oy >= 0 && oy < inputHeight && ox >= 0 && ox < inputWidth)
                        {
                            a += inputWeigths[((inputWidth * oy) + ox) * inputDepth + fd] * chainGradient;
                        }
                    }
                }

                filterGradients[depth * totalfilter + ((filterWidth * fy) + fx) * filterDepth + fd] = a;
            }
        }
}