using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace ConvNetSharp.Volume.GPU.Single
{
    public unsafe class VolumeStorage : VolumeStorage<float>, IDisposable
    {
        private readonly bool _isOwner;
        private readonly IntPtr _hostPointer;
        private bool _allocatedOnDevice;
        private readonly Layout _layout;

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
            var res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref this._hostPointer, this.Shape.TotalLength * sizeof(double));
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
            this.HostBuffer = (float*)this._hostPointer;

            // Zero out
            for (var i = 0; i < this.Shape.TotalLength; i++)
            {
                this.HostBuffer[i] = 0.0f;
            }

            this._isOwner = true;
            this._layout = new Layout(this.Shape.Dimensions); // TODO: not hardcode layout 
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

            this.Location = DataLocation.Host;
        }

        public VolumeStorage(VolumeStorage storage, Shape shape)
            : this(shape, storage.Context, storage.Shape.TotalLength)
        {
            this._isOwner = false;
            this.Location = storage.Location;
            this.HostBuffer = storage.HostBuffer;
            this._hostPointer = storage._hostPointer;
            this._allocatedOnDevice = storage._allocatedOnDevice;

            storage.CopyToDevice();
            this.DeviceBuffer = new CudaDeviceVariable<float>(storage.DeviceBuffer.DevicePointer);

            this.Location = DataLocation.Device;
        }

        public CudaDeviceVariable<byte> ConvolutionBackwardFilterStorage { get; set; }

        public CudaDeviceVariable<byte> ConvolutionBackwardStorage { get; set; }

        public CudaDeviceVariable<byte> ConvolutionStorage { get; set; }

        public DataLocation Location { get; set; }

        public float* HostBuffer { get; private set; }

        public CudaDeviceVariable<float> DeviceBuffer { get; private set; }

        public GpuContext Context { get; }

        public void Dispose() => Dispose(true);

        public override void Clear()
        {
            switch (this.Location)
            {
                case DataLocation.Host:
                    {
                        for (var i = 0; i < this.Shape.TotalLength; i++)
                        {
                            this.HostBuffer[i] = 0.0f;
                        }
                    }
                    break;
                case DataLocation.Device:
                    {
                        var res = DriverAPINativeMethods.Memset.cuMemsetD16_v2(this.DeviceBuffer.DevicePointer, 0, this.DeviceBuffer.Size * 2);
                        if (res != CUResult.Success)
                        {
                            throw new CudaException(res);
                        }
                    }
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public void CopyToDevice()
        {
            if (this.Location == DataLocation.Host)
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

                // Synchro
                this.Context.DefaultStream.Synchronize();

                this.Location = DataLocation.Device;
            }
        }

        public void CopyToHost()
        {
            if (this.Location == DataLocation.Device)
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

                this.Location = DataLocation.Host;
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

                if (this._isOwner)
                {
                    try
                    {
                        DriverAPINativeMethods.MemoryManagement.cuMemFreeHost(tmp);
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine(ex.Message);
                    }
                }
            }

            if (this._isOwner)
            {
                this.DeviceBuffer?.Dispose();
            }

            this.ConvolutionBackwardFilterStorage?.Dispose();
            this.ConvolutionBackwardStorage?.Dispose();
            this.ConvolutionStorage?.Dispose();
        }

        ~VolumeStorage()
        {
            Dispose(false);
        }

        public override float Get(int[] coordinates)
        {
            CopyToHost();

            var i = this._layout.IndexFromCoordinates(coordinates);
            return this.HostBuffer[i];
        }

        public override float Get(int w, int h, int c, int n)
        {
            CopyToHost();

            var i = this._layout.IndexFromCoordinates(w, h, c, n);
            return this.HostBuffer[i];
        }

        public override float Get(int w, int h, int c)
        {
            CopyToHost();

            var i = this._layout.IndexFromCoordinates(w, h, c);
            return this.HostBuffer[i];
        }

        public override float Get(int w, int h)
        {
            CopyToHost();

            var i = this._layout.IndexFromCoordinates(w, h);
            return this.HostBuffer[i];
        }

        public override float Get(int w)
        {
            CopyToHost();

            var i = this._layout.IndexFromCoordinates(w);
            return this.HostBuffer[i];
        }

        public override void Set(int[] coordinates, float value)
        {
            CopyToHost();

            var i = this._layout.IndexFromCoordinates(coordinates);
            this.HostBuffer[i] = value;
        }

        public override void Set(int w, int h, int c, int n, float value)
        {
            CopyToHost();

            var i = this._layout.IndexFromCoordinates(w, h, c, n);
            this.HostBuffer[i] = value;
        }

        public override void Set(int w, int h, int c, float value)
        {
            CopyToHost();

            var i = this._layout.IndexFromCoordinates(w, h, c);
            this.HostBuffer[i] = value;
        }

        public override void Set(int w, int h, float value)
        {
            CopyToHost();

            var i = this._layout.IndexFromCoordinates(w, h);
            this.HostBuffer[i] = value;
        }

        public override void Set(int w, float value)
        {
            CopyToHost();

            var i = this._layout.IndexFromCoordinates(w);
            this.HostBuffer[i] = value;
        }

        public override float[] ToArray()
        {
            CopyToHost();

            var array = new float[this.Shape.TotalLength];
            Marshal.Copy(new IntPtr(this.HostBuffer), array, 0, (int)this.Shape.TotalLength);
            return array;
        }

        public override bool Equals(VolumeStorage<float> other)
        {
            throw new NotImplementedException();
        }
    }
}