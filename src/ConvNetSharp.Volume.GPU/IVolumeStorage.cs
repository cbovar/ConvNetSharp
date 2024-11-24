using System.Numerics;
using ManagedCuda;

namespace ConvNetSharp.Volume.GPU;

public interface IVolumeStorage<T> where T : struct, INumber<T>, IMinMaxValue<T>
{
    CudaDeviceVariable<T> DeviceBuffer { get; }

    void CopyToDevice();
}