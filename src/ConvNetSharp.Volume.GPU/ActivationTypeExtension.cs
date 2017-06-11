using System;
using ManagedCuda.CudaDNN;

namespace ConvNetSharp.Volume.GPU
{
    internal static class ActivationTypeExtension
    {
        public static cudnnActivationMode ToCudnn(this ActivationType type)
        {
            switch (type)
            {
                case ActivationType.Sigmoid:
                    return cudnnActivationMode.Sigmoid;
                case ActivationType.Relu:
                    return cudnnActivationMode.Relu;
                case ActivationType.Tanh:
                    return cudnnActivationMode.Tanh;
                case ActivationType.ClippedRelu:
                    return cudnnActivationMode.ClippedRelu;
            }

            throw new NotImplementedException();
        }
    }
}