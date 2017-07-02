using System;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace ConvNetSharp.Volume.GPU
{
    internal class CudaHostMemoryRegion : IDisposable
    {
        private readonly IntPtr _startPointer;
        private bool _releasedCudaMemory;

        public CudaHostMemoryRegion(long byteCount)
        {
            if (byteCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(byteCount));
            }

            this._releasedCudaMemory = true;
            this.ByteCount = byteCount;
            var result = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref this._startPointer, byteCount);

            if (result != CUResult.Success)
            {
                throw new CudaException(result);
            }

            this._releasedCudaMemory = false;
        }

        public IntPtr Start => this._startPointer;

        public long ByteCount { get; }

        public void Dispose()
        {
            ReleaseUnmanagedResources();
            GC.SuppressFinalize(this);
        }

        ~CudaHostMemoryRegion()
        {
            ReleaseUnmanagedResources();
        }

        private void ReleaseUnmanagedResources()
        {
            if (this._releasedCudaMemory)
            {
                return;
            }

            var result = DriverAPINativeMethods.MemoryManagement.cuMemFreeHost(this._startPointer);
            if (result != CUResult.Success)
            {
                throw new CudaException(result);
            }

            this._releasedCudaMemory = true;
        }
    }
}