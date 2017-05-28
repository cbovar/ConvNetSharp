using System;
using ManagedCuda;
using ManagedCuda.CudaDNN;

namespace ConvNetSharp.Volume.GPU.Double
{
    public class Volume : Volume<double>
    {
        private readonly GpuContext _context;
        private readonly VolumeStorage _volumeStorage;

        public Volume(VolumeStorage storage) : base(storage)
        {
            this._context = storage.Context;
            this._volumeStorage = this.Storage as VolumeStorage;
        }

        public Volume(double[] array, Shape shape) : base(new VolumeStorage(array, shape, GpuContext.Default))
        {
            this._context = GpuContext.Default;
            this._volumeStorage = this.Storage as VolumeStorage;
        }

        public Volume(double[] array, Shape shape, GpuContext context) : base(new VolumeStorage(array, shape, context))
        {
            this._context = context;
            this._volumeStorage = this.Storage as VolumeStorage;
        }

        public override void DoAdd(Volume<double> other, Volume<double> result)
        {
            var otherStorage = other.Storage as VolumeStorage;
            var resultStorage = result.Storage as VolumeStorage;

            if (otherStorage == null)
            {
                throw new ArgumentException($"{nameof(other)} storage should be VolumeStorage", nameof(other));
            }

            if (resultStorage == null)
            {
                throw new ArgumentException($"{nameof(result)} storage should be VolumeStorage", nameof(result));
            }

            // Copy to device if not already done
            this._volumeStorage.CopyToDevice();
            otherStorage.CopyToDevice();
            resultStorage.CopyToDevice();

            // result = this
            DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy(resultStorage.DeviceBuffer.DevicePointer,
                this._volumeStorage.DeviceBuffer.DevicePointer, this.Shape.TotalLength * sizeof(double));

            // Synchro
            this._context.DefaultStream.Synchronize();

            // Add tensors
            using (var biasDesc = new TensorDescriptor())
            using (var srcDesc = new TensorDescriptor())
            {
                srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double,
                    this.Shape.GetDimension(3),
                    this.Shape.GetDimension(2),
                    this.Shape.GetDimension(1),
                    this.Shape.GetDimension(0));

                biasDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double,
                    other.Shape.GetDimension(3),
                    other.Shape.GetDimension(2),
                    other.Shape.GetDimension(1),
                    other.Shape.GetDimension(0));

                this._context.CudnnContext.AddTensor(1.0,
                    biasDesc, otherStorage.DeviceBuffer, 1.0,
                    srcDesc, resultStorage.DeviceBuffer);
            }
        }

        public override void DoNegate(Volume<double> result)
        {
            DoMultiply(-1.0, result);
        }

        public override void DoMultiply(Volume<double> other, Volume<double> result)
        {
            if (other.Shape.TotalLength == 1)
            {
                var resultStorage = result.Storage as VolumeStorage;
                if (resultStorage == null)
                {
                    throw new ArgumentException($"{nameof(result)} storage should be VolumeStorage", nameof(result));
                }
                // Copy to device if not already done
                this._volumeStorage.CopyToDevice();
                resultStorage.CopyToDevice();

                // result = this
                DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy(resultStorage.DeviceBuffer.DevicePointer,
                    this._volumeStorage.DeviceBuffer.DevicePointer, this.Shape.TotalLength * sizeof(double));

                // Synchro
                this._context.DefaultStream.Synchronize();

                // Add tensors
                using (var srcDesc = new TensorDescriptor())
                {
                    var n = result.Shape.GetDimension(3);
                    var c = result.Shape.GetDimension(2);
                    var h = result.Shape.GetDimension(1);
                    var w = result.Shape.GetDimension(0);

                    srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double, n, c, h, w);

                    this._context.CudnnContext.ScaleTensor(srcDesc, resultStorage.DeviceBuffer, other.Get(0));
                }
            }
            else
            {
                throw new NotImplementedException();
            }
        }
    }
}