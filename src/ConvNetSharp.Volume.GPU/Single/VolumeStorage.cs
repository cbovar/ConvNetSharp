using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace ConvNetSharp.Volume.GPU.Single
{
    public unsafe class VolumeStorage : VolumeStorage<float>, IDisposable
    {
        private readonly IntPtr _hostPointer;
        private bool _allocatedOnDevice;

        public VolumeStorage(Shape shape, GpuContext context, long length = -1) : base(shape)
        {
            this.Context = context;

            // Take care of unkown dimension
            if (length != -1)
            {
                this.Shape.GuessUnkownDimension(length);
            }

            // Host 
            this._hostPointer = IntPtr.Zero;
            var res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref this._hostPointer, this.Shape.TotalLength * sizeof(float));
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
            this.HostBuffer = (float*) this._hostPointer;
        }

        public VolumeStorage(float[] array, Shape shape, GpuContext context) : this(shape, context, array.Length)
        {
            this.Context = context;

            if (this.Shape.TotalLength != array.Length)
            {
                throw new ArgumentException("Wrong dimensions");
            }

            // Fill host buffer
            for (var i = 0; i < array.Length; i++)
            {
                this.HostBuffer[i] = array[i];
            }
        }

        public VolumeStorage(VolumeStorage storage, Shape shape)
            : this(shape, storage.Context, storage.Shape.TotalLength)
        {
            // Ensure it is on host
            storage.CopyToHost();

            // Fill host buffer
            for (var i = 0; i < this.Shape.TotalLength; i++)
            {
                this.HostBuffer[i] = storage.HostBuffer[i];
            }
        }

        public CudaDeviceVariable<byte> ConvolutionBackwardFilterStorage { get; set; }

        public CudaDeviceVariable<byte> ConvolutionBackwardStorage { get; set; }

        public CudaDeviceVariable<byte> ConvolutionStorage { get; set; }

        public bool CopiedToDevice { get; set; }

        public bool CopiedToHost { get; set; }

        public float* HostBuffer { get; private set; }

        public CudaDeviceVariable<float> DeviceBuffer { get; private set; }

        public GpuContext Context { get; }

        public void Dispose() => Dispose(true);

        public override void Clear()
        {
            CopyToDevice();

            DriverAPINativeMethods.Memset.cuMemsetD16_v2(this.DeviceBuffer.DevicePointer, 0, this.DeviceBuffer.SizeInBytes);

            this.CopiedToDevice = true;
            this.CopiedToHost = false;
        }

        public void CopyToDevice()
        {
            if (!this.CopiedToDevice)
            {
                // Device 
                if (!this._allocatedOnDevice)
                {
                    this.DeviceBuffer = new CudaDeviceVariable<float>(this.Shape.TotalLength);
                    this._allocatedOnDevice = true;
                }

                var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(
                    this.DeviceBuffer.DevicePointer, this._hostPointer, this.DeviceBuffer.SizeInBytes,
                    this.Context.DefaultStream.Stream);
                if (res != CUResult.Success)
                {
                    throw new CudaException(res);
                }
            }

            this.CopiedToDevice = true;
            this.CopiedToHost = false;
        }

        public void CopyToHost()
        {
            if (this.CopiedToDevice && !this.CopiedToHost)
            {
                var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(
                    new IntPtr(this.HostBuffer),
                    this.DeviceBuffer.DevicePointer, this.DeviceBuffer.SizeInBytes, this.Context.DefaultStream.Stream);

                if (res != CUResult.Success)
                {
                    throw new CudaException(res);
                }

                // Synchro
                this.Context.DefaultStream.Synchronize();

                this.CopiedToHost = true;
            }
        }

        public virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                GC.SuppressFinalize(this);
            }

            if (this.HostBuffer != default(float*))
            {
                var tmp = new IntPtr(this.HostBuffer);
                this.HostBuffer = default(float*);
                try
                {
                    DriverAPINativeMethods.MemoryManagement.cuMemFreeHost(tmp);
                }
                catch (Exception ex)
                {
                    Debug.WriteLine(ex.Message);
                }
            }

            this.DeviceBuffer?.Dispose();
            this.ConvolutionBackwardFilterStorage?.Dispose();
            this.ConvolutionBackwardStorage?.Dispose();
            this.ConvolutionStorage?.Dispose();
        }

        ~VolumeStorage()
        {
            Dispose(false);
        }

        public override float Get(int w, int h, int c, int n)
        {
            CopyToHost();

            return this.HostBuffer[
                w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) +
                n * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) * this.Shape.GetDimension(2)];
        }

        public override float Get(int w, int h, int c)
        {
            CopyToHost();
            return
                this.HostBuffer[
                    w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1)];
        }

        public override float Get(int w, int h)
        {
            CopyToHost();
            return this.HostBuffer[w + h * this.Shape.GetDimension(0)];
        }

        public override float Get(int i)
        {
            CopyToHost();
            return this.HostBuffer[i];
        }

        public override void Set(int w, int h, int c, int n, float value)
        {
            CopyToHost();
            this.HostBuffer[
                w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) +
                n * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) * this.Shape.GetDimension(2)] = value;
            this.CopiedToDevice = false;
        }

        public override void Set(int w, int h, int c, float value)
        {
            CopyToHost();
            this.HostBuffer[
                    w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1)] =
                value;
            this.CopiedToDevice = false;
        }

        public override void Set(int w, int h, float value)
        {
            CopyToHost();
            this.HostBuffer[w + h * this.Shape.GetDimension(0)] = value;
            this.CopiedToDevice = false;
        }

        public override void Set(int i, float value)
        {
            CopyToHost();
            this.HostBuffer[i] = value;
            this.CopiedToDevice = false;
        }

        public override float[] ToArray()
        {
            CopyToHost();

            var array = new float[this.Shape.TotalLength];
            Marshal.Copy(new IntPtr(this.HostBuffer), array, 0, (int) this.Shape.TotalLength);
            return array;
        }
    }
}