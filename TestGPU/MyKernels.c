typedef struct {
    int Id;
    unsigned int Value;
} SomeBasicType;

extern "C" {
    __global__ void Multiply(const int N, SomeBasicType* __restrict data, int factor) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
            i += blockDim.x * gridDim.x)
        {
            (data + i)->Value *= factor;
        }
    }
}