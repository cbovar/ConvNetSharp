using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace ConvNetSharp.Volume.GPU.Double
{
    public unsafe class VolumeStorage : VolumeStorage<double>, IDisposable
    {
        private readonly IntPtr _hostPointer;
        private readonly bool _isOwner;
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
            var res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref this._hostPointer, this.Shape.TotalLength * sizeof(double));
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
            this.HostBuffer = (double*)this._hostPointer;

            // Zero out
            for (var i = 0; i < this.Shape.TotalLength; i++)
            {
                this.HostBuffer[i] = 0.0;
            }

            this._isOwner = true;
        }

        public VolumeStorage(double[] array, Shape shape, GpuContext context) : this(shape, context, array.Length)
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
            this.DeviceBuffer = new CudaDeviceVariable<double>(storage.DeviceBuffer.DevicePointer);

            this.Location = DataLocation.Device;
        }

        public CudaDeviceVariable<byte> ConvolutionBackwardFilterStorage { get; set; }

        public CudaDeviceVariable<byte> ConvolutionBackwardStorage { get; set; }

        public CudaDeviceVariable<byte> ConvolutionStorage { get; set; }

        public DataLocation Location { get; set; }

        public double* HostBuffer { get; private set; }

        public CudaDeviceVariable<double> DeviceBuffer { get; private set; }

        public GpuContext Context { get; }

        public void Dispose()
        {
            Dispose(true);
        }

        public override void CopyFrom(VolumeStorage<double> source)
        {
            var real = source as VolumeStorage;

            if (!object.ReferenceEquals(this, real))
            {
                if (this.Shape.TotalLength != real.Shape.TotalLength)
                    throw new ArgumentException($"{nameof(real)} has different length!");

                real.CopyToDevice();

                if (!this._allocatedOnDevice)
                {
                    this.DeviceBuffer = new CudaDeviceVariable<double>(this.Shape.TotalLength);
                    this._allocatedOnDevice = true;
                }

                var res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy(
                    this.DeviceBuffer.DevicePointer,
                    real.DeviceBuffer.DevicePointer,
                    this.Shape.TotalLength*sizeof(double));

                if (res != CUResult.Success)
                    throw new CudaException(res);

                this.Location = DataLocation.Device;
            }
            else
            {
                this.CopyToDevice();
            }
        }

        public override void Clear()
        {
            switch (this.Location)
            {
                case DataLocation.Host:
                    {
                        for (var i = 0; i < this.Shape.TotalLength; i++)
                        {
                            this.HostBuffer[i] = 0.0;
                        }
                    }
                    break;
                case DataLocation.Device:
                    {
                        var res = DriverAPINativeMethods.Memset.cuMemsetD32_v2(this.DeviceBuffer.DevicePointer, 0, this.DeviceBuffer.Size * 2);
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
                    this.DeviceBuffer = new CudaDeviceVariable<double>(this.Shape.TotalLength);
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

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                GC.SuppressFinalize(this);
            }

            if (this.HostBuffer != default(double*))
            {
                var tmp = new IntPtr(this.HostBuffer);
                this.HostBuffer = default(double*);

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

        public override double Get(int[] coordinates)
        {
            CopyToHost();

            var length = coordinates.Length;
            return Get(coordinates[0], length > 1 ? coordinates[1] : 0, length > 2 ? coordinates[2] : 0, length > 3 ? coordinates[3] : 0);
        }

        public override double Get(int w, int h, int c, int n)
        {
            CopyToHost();

            return this.HostBuffer[
                w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) +
                n * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) * this.Shape.GetDimension(2)];
        }

        public override double Get(int w, int h, int c)
        {
            CopyToHost();
            return
                this.HostBuffer[
                    w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1)];
        }

        public override double Get(int w, int h)
        {
            CopyToHost();
            return this.HostBuffer[w + h * this.Shape.GetDimension(0)];
        }

        public override double Get(int i)
        {
            CopyToHost();
            return this.HostBuffer[i];
        }

        public override void Set(int[] coordinates, double value)
        {
            CopyToHost();

            var length = coordinates.Length;
            Set(coordinates[0], length > 1 ? coordinates[1] : 0, length > 2 ? coordinates[2] : 0, length > 3 ? coordinates[3] : 0, value);
        }

        public override void Set(int w, int h, int c, int n, double value)
        {
            CopyToHost();
            this.HostBuffer[
                w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) +
                n * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) * this.Shape.GetDimension(2)] = value;
        }

        public override void Set(int w, int h, int c, double value)
        {
            CopyToHost();
            this.HostBuffer[
                    w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1)] =
                value;
        }

        public override void Set(int w, int h, double value)
        {
            CopyToHost();
            this.HostBuffer[w + h * this.Shape.GetDimension(0)] = value;
        }

        public override void Set(int i, double value)
        {
            CopyToHost();
            this.HostBuffer[i] = value;
        }

        public override double[] ToArray()
        {
            CopyToHost();

            var array = new double[this.Shape.TotalLength];
            Marshal.Copy(new IntPtr(this.HostBuffer), array, 0, (int)this.Shape.TotalLength);
            return array;
        }
    }
}