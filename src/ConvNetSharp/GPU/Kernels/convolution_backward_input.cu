extern "C" {

 __global__ void Run(int pad, int stride,
            int filterWidth, int filterHeight, int filterDepth, double* __restrict filterWeights,
            int inputWidth, int inputHeight, int inputDepth, double* __restrict intputGradients,
            int outputWidth, int outputHeight, int outputDepth, double* __restrict outputGradients)
        {
            int t = blockIdx.x * blockDim.x + threadIdx.x;
            int total = inputWidth * inputHeight;
            int temp = t % total;
            int fd = t / total;
            int ox = temp % inputWidth;
            int oy = temp / inputWidth;

            if (fd < filterDepth && ox < inputWidth && oy < inputHeight)
            {
                int totalfilter = filterWidth * filterHeight * filterDepth;

                double a = 0.0;

                for (int fy = 0; fy < filterHeight; fy++)
                {
                    int y = oy - fy + pad;

                    if (y % stride == 0)
                    {
                        int ay = y / stride;

                        if (ay >= 0 && ay < outputHeight)
                        {
                            for (int fx = 0; fx < filterWidth; fx++)
                            {
                                int x = ox - fx + pad;

                                if (x % stride == 0)
                                {
                                    int ax = x / stride;

                                    if (ax >= 0 && ax < outputWidth)
                                    {
                                        int a0 = ((filterWidth * fy) + fx) * filterDepth + fd;
                                        int ix0 = ((outputWidth * ay) + ax) * outputDepth;

                                        for (int depth = 0; depth < outputDepth; depth++)
                                        {
                                            int ix = ix0 + depth;
                                            double chainGradient = outputGradients[ix];

                                            a += filterWeights[depth * totalfilter + a0] * chainGradient;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                intputGradients[((inputWidth * oy) + ox) * inputDepth + fd] = a;
            }
        }
}