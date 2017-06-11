using System;
using ManagedCuda;
using ManagedCuda.BasicTypes;
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

        private void DoActivation(Volume<double> result, cudnnActivationMode mode)
        {
            var resultStorage = result.Storage as VolumeStorage;
            if (resultStorage == null)
            {
                throw new ArgumentException($"{nameof(result)} storage should be VolumeStorage", nameof(result));
            }

            // Copy to device if not already done
            this._volumeStorage.CopyToDevice();
            resultStorage.CopyToDevice();

            // Synchro
            this._context.DefaultStream.Synchronize();

            // Relu
            using (var activationDesc = new ActivationDescriptor())
            using (var srcDesc = new TensorDescriptor())
            using (var resultDesc = new TensorDescriptor())
            {
                var n = result.Shape.GetDimension(3);
                var c = result.Shape.GetDimension(2);
                var h = result.Shape.GetDimension(1);
                var w = result.Shape.GetDimension(0);

                srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double, n, c, h, w);
                resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double, n, c, h, w);
                activationDesc.SetActivationDescriptor(mode, cudnnNanPropagation.NotPropagateNan, 0.0);

                this._context.CudnnContext.ActivationForward(activationDesc, 1.0, srcDesc, this._volumeStorage.DeviceBuffer, 0.0,
                    resultDesc, resultStorage.DeviceBuffer);
            }
        }

        public override void DoActivation(Volume<double> result, ActivationType type)
        {
            DoActivation(result, type.ToCudnn());
        }

        private void DoActivationGradient(Volume<double> input, Volume<double> outputGradient, Volume<double> inputGradient, cudnnActivationMode mode)
        {
            var inputStorage = input.Storage as VolumeStorage;
            var inputGradientStorage = inputGradient.Storage as VolumeStorage;
            var outputStorage = this._volumeStorage;
            var outputGradientStorage = outputGradient.Storage as VolumeStorage;

            // Copy to device if not already done
            outputStorage.CopyToDevice();
            outputGradientStorage.CopyToDevice();
            inputGradientStorage.CopyToDevice();

            // Synchro
            this._context.DefaultStream.Synchronize();

            using (var activationDesc = new ActivationDescriptor())
            using (var srcDesc = new TensorDescriptor())
            using (var srcDiffDesc = new TensorDescriptor())
            using (var destDesc = new TensorDescriptor())
            using (var destDiffDesc = new TensorDescriptor())
            {
                var n = this.Shape.GetDimension(3);
                var c = this.Shape.GetDimension(2);
                var h = this.Shape.GetDimension(1);
                var w = this.Shape.GetDimension(0);

                srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double, n, c, h, w);
                srcDiffDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double, n, c, h, w);
                destDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double, n, c, h, w);
                destDiffDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double, n, c, h, w);

                activationDesc.SetActivationDescriptor(mode, cudnnNanPropagation.NotPropagateNan,
                    0.0);

                this._context.CudnnContext.ActivationBackward(activationDesc, 1.0,
                    srcDesc, outputStorage.DeviceBuffer,
                    srcDiffDesc, outputGradientStorage.DeviceBuffer,
                    destDesc, inputStorage.DeviceBuffer,
                    0.0,
                    destDiffDesc, inputGradientStorage.DeviceBuffer);
            }
        }

        public override void DoActivationGradient(Volume<double> input, Volume<double> outputGradient, Volume<double> result, ActivationType type)
        {
            DoActivationGradient(input, outputGradient, result, type.ToCudnn());
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

        public override void DoConvolution(Volume<double> filters, int pad, int stride, Volume<double> result)
        {
            var resultStorage = result.Storage as VolumeStorage;
            if (resultStorage == null)
            {
                throw new ArgumentException($"{nameof(result)} storage should be VolumeStorage", nameof(result));
            }

            var inputStorage = this._volumeStorage;
            var filterStorage = filters.Storage as VolumeStorage;

            // Copy to device if not already done
            inputStorage.CopyToDevice();
            filterStorage.CopyToDevice();
            resultStorage.CopyToDevice();

            // Synchro
            this._context.DefaultStream.Synchronize();

            using (var dataDesc = new TensorDescriptor())
            using (var filterDesc = new FilterDescriptor())
            using (var outputDesc = new TensorDescriptor())
            using (var convolutionDesc = new ConvolutionDescriptor())
            {
                convolutionDesc.SetConvolution2dDescriptor(pad, pad, stride, stride, 1, 1,
                    cudnnConvolutionMode.CrossCorrelation, cudnnDataType.Double);

                dataDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double,
                    this.Shape.GetDimension(3),
                    this.Shape.GetDimension(2),
                    this.Shape.GetDimension(1),
                    this.Shape.GetDimension(0));

                filterDesc.SetFilter4dDescriptor(cudnnDataType.Double, cudnnTensorFormat.NCHW,
                    filters.Shape.GetDimension(3),
                    filters.Shape.GetDimension(2),
                    filters.Shape.GetDimension(1),
                    filters.Shape.GetDimension(0));

                outputDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double,
                    result.Shape.GetDimension(3),
                    result.Shape.GetDimension(2),
                    result.Shape.GetDimension(1),
                    result.Shape.GetDimension(0));

                var algo = this._context.CudnnContext.GetConvolutionForwardAlgorithm(
                    dataDesc, filterDesc,
                    convolutionDesc, outputDesc,
                    cudnnConvolutionFwdPreference.PreferFastest, IntPtr.Zero);

                var workspaceSize = this._context.CudnnContext.GetConvolutionForwardWorkspaceSize(
                    dataDesc, filterDesc,
                    convolutionDesc, outputDesc, algo);
                workspaceSize = workspaceSize == 0 ? new SizeT(1) : workspaceSize;

                if (this._volumeStorage.ConvolutionStorage == null || this._volumeStorage.ConvolutionStorage.Size != workspaceSize)
                {
                    this._volumeStorage.ConvolutionStorage = new CudaDeviceVariable<byte>(workspaceSize);
                }

                this._context.CudnnContext.ConvolutionForward(1.0,
                    dataDesc, inputStorage.DeviceBuffer,
                    filterDesc, filterStorage.DeviceBuffer,
                    convolutionDesc, algo, this._volumeStorage.ConvolutionStorage, 0.0,
                    outputDesc, resultStorage.DeviceBuffer);
            }
        }

        public override void DoConvolutionGradient(Volume<double> filters, Volume<double> outputGradients,
            Volume<double> inputGradient, Volume<double> filterGradient, int pad,
            int stride)
        {
            var inputStorage = this._volumeStorage;
            var outputGradientStorage = outputGradients.Storage as VolumeStorage;
            var filterstorage = filters.Storage as VolumeStorage;
            var inputGradientStorage = inputGradient.Storage as VolumeStorage;
            var filterGradientStorage = filterGradient.Storage as VolumeStorage;

            // Copy to device if not already done
            inputStorage.CopyToDevice();
            outputGradientStorage.CopyToDevice();
            filterstorage.CopyToDevice();
            inputGradientStorage.CopyToDevice();
            filterGradientStorage.CopyToDevice();

            using (var dataDesc = new TensorDescriptor())
            using (var filterDesc = new FilterDescriptor())
            using (var dDataDesc = new TensorDescriptor())
            using (var dOutputDesc = new TensorDescriptor())
            using (var dfilterDesc = new FilterDescriptor())
            using (var convolutionDesc = new ConvolutionDescriptor())
            {
                convolutionDesc.SetConvolution2dDescriptor(pad, pad, stride, stride, 1, 1,
                    cudnnConvolutionMode.CrossCorrelation, cudnnDataType.Double);

                dataDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double,
                    this.Shape.GetDimension(3),
                    this.Shape.GetDimension(2),
                    this.Shape.GetDimension(1),
                    this.Shape.GetDimension(0));

                dDataDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double,
                    this.Shape.GetDimension(3),
                    this.Shape.GetDimension(2),
                    this.Shape.GetDimension(1),
                    this.Shape.GetDimension(0));

                dOutputDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double,
                    outputGradients.Shape.GetDimension(3),
                    outputGradients.Shape.GetDimension(2),
                    outputGradients.Shape.GetDimension(1),
                    outputGradients.Shape.GetDimension(0));

                filterDesc.SetFilter4dDescriptor(cudnnDataType.Double, cudnnTensorFormat.NCHW,
                    filters.Shape.GetDimension(3),
                    filters.Shape.GetDimension(2),
                    filters.Shape.GetDimension(1),
                    filters.Shape.GetDimension(0));

                dfilterDesc.SetFilter4dDescriptor(cudnnDataType.Double, cudnnTensorFormat.NCHW,
                    filters.Shape.GetDimension(3),
                    filters.Shape.GetDimension(2),
                    filters.Shape.GetDimension(1),
                    filters.Shape.GetDimension(0));

                var filterAlgo = this._context.CudnnContext.GetConvolutionBackwardFilterAlgorithm(dataDesc, dOutputDesc,
                    convolutionDesc, dfilterDesc, cudnnConvolutionBwdFilterPreference.PreferFastest, IntPtr.Zero);
                var filterWorkspaceSize = this._context.CudnnContext.GetConvolutionBackwardFilterWorkspaceSize(dataDesc,
                    dOutputDesc, convolutionDesc, dfilterDesc, filterAlgo);
                filterWorkspaceSize = filterWorkspaceSize == 0 ? new SizeT(1) : filterWorkspaceSize;

                var dataAlgo = this._context.CudnnContext.GetConvolutionBackwardDataAlgorithm(filterDesc, dOutputDesc,
                    convolutionDesc, dDataDesc, cudnnConvolutionBwdDataPreference.PreferFastest, IntPtr.Zero);
                var dataWorkspaceSize = this._context.CudnnContext.GetConvolutionBackwardDataWorkspaceSize(dfilterDesc,
                    dOutputDesc, convolutionDesc, dDataDesc, dataAlgo);
                dataWorkspaceSize = dataWorkspaceSize == 0 ? new SizeT(1) : dataWorkspaceSize;

                // filter
                if (this._volumeStorage.ConvolutionBackwardFilterStorage == null || this._volumeStorage.ConvolutionBackwardFilterStorage.Size != filterWorkspaceSize)
                {
                    this._volumeStorage.ConvolutionBackwardFilterStorage = new CudaDeviceVariable<byte>(filterWorkspaceSize);
                }
                this._context.CudnnContext.ConvolutionBackwardFilter(1.0, dataDesc, inputStorage.DeviceBuffer, dOutputDesc,
                    outputGradientStorage.DeviceBuffer, convolutionDesc, filterAlgo,
                    this._volumeStorage.ConvolutionBackwardFilterStorage, 0.0, dfilterDesc,
                    filterGradientStorage.DeviceBuffer);

                // data
                if (this._volumeStorage.ConvolutionBackwardStorage == null || this._volumeStorage.ConvolutionBackwardStorage.Size != dataWorkspaceSize)
                {
                    this._volumeStorage.ConvolutionBackwardStorage = new CudaDeviceVariable<byte>(dataWorkspaceSize);
                }
                this._context.CudnnContext.ConvolutionBackwardData(1.0, filterDesc, filterstorage.DeviceBuffer, dOutputDesc,
                    outputGradientStorage.DeviceBuffer, convolutionDesc, dataAlgo,
                    this._volumeStorage.ConvolutionBackwardStorage, 0.0, dDataDesc,
                    inputGradientStorage.DeviceBuffer);
            }
        }

        public override void DoDivide(Volume<double> other, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoLog(Volume<double> result)
        {
            throw new NotImplementedException();
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

        public override void DoNegate(Volume<double> result)
        {
            DoMultiply(-1.0, result);
        }

        public override void DoSoftmax(Volume<double> result)
        {
            var inputStorage = this._volumeStorage;
            var outputStorage = result.Storage as VolumeStorage;

            // Copy to device if not already done
            inputStorage.CopyToDevice();
            outputStorage.CopyToDevice();

            using (var srcDesc = new TensorDescriptor())
            using (var destDesc = new TensorDescriptor())
            {
                var n = this.Shape.GetDimension(3);
                var c = this.Shape.GetDimension(2);
                var h = this.Shape.GetDimension(1);
                var w = this.Shape.GetDimension(0);

                srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double, n, c, h, w);
                destDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double, n, c, h, w);

                this._context.CudnnContext.SoftmaxForward(cudnnSoftmaxAlgorithm.Accurate, cudnnSoftmaxMode.Channel, 1.0,
                    srcDesc, inputStorage.DeviceBuffer, 0.0,
                    destDesc, outputStorage.DeviceBuffer);
            }
        }

        public override void DoSoftmaxGradient(Volume<double> y, Volume<double> inputGradient)
        {
            var inputGradientStorage = (VolumeStorage)inputGradient.Storage;
            var outputGradientStorage = (VolumeStorage)y.Storage;
            var outputStorage = this._volumeStorage;

            // Copy to device if not already done
            outputStorage.CopyToDevice();
            outputGradientStorage.CopyToDevice();
            inputGradientStorage.CopyToDevice();

            // Synchro
            this._context.DefaultStream.Synchronize();

            using (var activationDesc = new ActivationDescriptor())
            using (var srcDesc = new TensorDescriptor())
            using (var srcDiffDesc = new TensorDescriptor())
            using (var destDiffDesc = new TensorDescriptor())
            {
                var n = this.Shape.GetDimension(3);
                var c = this.Shape.GetDimension(2);
                var h = this.Shape.GetDimension(1);
                var w = this.Shape.GetDimension(0);

                srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double, n, c, h, w);
                srcDiffDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double, n, c, h, w);
                destDiffDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double, n, c, h, w);
                activationDesc.SetActivationDescriptor(cudnnActivationMode.Relu, cudnnNanPropagation.PropagateNan, 0.0);

                this._context.CudnnContext.SoftmaxBackward(cudnnSoftmaxAlgorithm.Accurate, cudnnSoftmaxMode.Channel, 1.0,
                    srcDesc, outputStorage.DeviceBuffer,
                    srcDiffDesc, outputGradientStorage.DeviceBuffer,
                    0.0,
                    destDiffDesc, inputGradientStorage.DeviceBuffer);
            }
        }

        public override void DoExp(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoMax(Volume<double> result)
        {
            throw new NotImplementedException();

            //    var inputStorage = this._volumeStorage;
            //    var outputStorage = result.Storage as VolumeStorage;

            //    // Copy to device if not already done
            //    inputStorage.CopyToDevice();
            //    outputStorage.CopyToDevice();

            //    using (var opDesc = new OpTensorDescriptor(this._context.CudnnContext))
            //    using (var srcDesc = new TensorDescriptor())
            //    using (var destDesc = new TensorDescriptor())
            //    {
            //        var n = this.Shape.GetDimension(3);
            //        var c = this.Shape.GetDimension(2);
            //        var h = this.Shape.GetDimension(1);
            //        var w = this.Shape.GetDimension(0);

            //        srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double, n, c, h, w);
            //        destDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Double, n, c, h, w);
            //        opDesc.SetOpTensorDescriptor(cudnnOpTensorOp.OpTensorMax, cudnnDataType.Double, cudnnNanPropagation.NotPropagateNan);

            //        this._context.CudnnContext.OpTensor(opDesc, 1.0, srcDesc, inputStorage.DeviceBuffer.)

            //        this._context.CudnnContext.SoftmaxForward(cudnnSoftmaxAlgorithm.Accurate, cudnnSoftmaxMode.Channel, 1.0,
            //            srcDesc, inputStorage.DeviceBuffer, 0.0,
            //            destDesc, outputStorage.DeviceBuffer);
            //    }
        }
    }
}