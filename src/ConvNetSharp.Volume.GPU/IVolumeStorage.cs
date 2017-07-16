using System;
using ManagedCuda;

namespace ConvNetSharp.Volume.GPU
{
    public interface IVolumeStorage<T> where T : struct, IEquatable<T>, IFormattable
    {
        CudaDeviceVariable<T> DeviceBuffer { get; }

        void CopyToDevice();
    }
}