using System;
using System.IO;
using System.Reflection;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaDNN;

namespace ConvNetSharp.Volume.GPU.Single
{
    public class Volume : Volume<float>
    {
        private static KernelLoader<float> _kernelLoader;
        private readonly GpuContext _context;
        private readonly VolumeStorage _volumeStorage;

        internal Volume(VolumeStorage storage) : base(storage)
        {
            this._context = storage.Context;
            this._volumeStorage = this.Storage as VolumeStorage;

            LoadKernels();
        }

        internal Volume(float[] array, Shape shape) : base(new VolumeStorage(array, shape, GpuContext.Default))
        {
            this._context = GpuContext.Default;
            this._volumeStorage = this.Storage as VolumeStorage;

            LoadKernels();
        }

        public Volume(float[] array, Shape shape, GpuContext context) : base(new VolumeStorage(array, shape, context))
        {
            this._context = context;
            this._volumeStorage = this.Storage as VolumeStorage;

            LoadKernels();
        }

        private void DoActivation(Volume<float> result, cudnnActivationMode mode)
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

                srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
                resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
                activationDesc.SetActivationDescriptor(mode, cudnnNanPropagation.NotPropagateNan, 0.0);

                this._context.CudnnContext.ActivationForward(activationDesc,
                    1.0f, srcDesc, this._volumeStorage.DeviceBuffer,
                    0.0f, resultDesc, resultStorage.DeviceBuffer);
            }
        }

        public override void DoActivation(Volume<float> result, ActivationType type)
        {
            DoActivation(result, type.ToCudnn());
        }

        private void DoActivationGradient(Volume<float> input, Volume<float> outputGradient, Volume<float> inputGradient, cudnnActivationMode mode)
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

                srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
                srcDiffDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
                destDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
                destDiffDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);

                activationDesc.SetActivationDescriptor(mode, cudnnNanPropagation.NotPropagateNan,
                    0.0);

                this._context.CudnnContext.ActivationBackward(activationDesc, 1.0f,
                    srcDesc, outputStorage.DeviceBuffer,
                    srcDiffDesc, outputGradientStorage.DeviceBuffer,
                    destDesc, inputStorage.DeviceBuffer,
                    0.0f,
                    destDiffDesc, inputGradientStorage.DeviceBuffer);
            }
        }

        public override void DoActivationGradient(Volume<float> input, Volume<float> outputGradient,
            Volume<float> result, ActivationType type)
        {
            DoActivationGradient(input, outputGradient, result, type.ToCudnn());
        }

        public override void DoAdd(Volume<float> other, Volume<float> result)
        {
            if (ReferenceEquals(other, result))
            {
                throw new NotSupportedException("other and result should not be the same!");
            }

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
            resultStorage.CopyFrom(this._volumeStorage);
            otherStorage.CopyToDevice();

            // Add tensors
            using (var otherDesc = new TensorDescriptor())
            using (var resultDesc = new TensorDescriptor())
            {
                resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    result.Shape.GetDimension(3),
                    result.Shape.GetDimension(2),
                    result.Shape.GetDimension(1),
                    result.Shape.GetDimension(0));

                otherDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    other.Shape.GetDimension(3),
                    other.Shape.GetDimension(2),
                    other.Shape.GetDimension(1),
                    other.Shape.GetDimension(0));

                this._context.CudnnContext.AddTensor(
                    1.0f, otherDesc, otherStorage.DeviceBuffer,
                    1.0f, resultDesc, resultStorage.DeviceBuffer);
            }
        }

        protected override void DoBiasGradient(Volume<float> biasGradient)
        {
            var outputGradientStorage = this._volumeStorage;
            var biasGradientStorage = biasGradient.Storage as VolumeStorage;

            // Copy to device if not already done
            outputGradientStorage.CopyToDevice();
            biasGradientStorage.CopyToDevice();

            using (var dOutputDesc = new TensorDescriptor())
            using (var dBiasDesc = new TensorDescriptor())
            {
                dOutputDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    this.Shape.GetDimension(3),
                    this.Shape.GetDimension(2),
                    this.Shape.GetDimension(1),
                    this.Shape.GetDimension(0));

                dBiasDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    biasGradient.Shape.GetDimension(3),
                    biasGradient.Shape.GetDimension(2),
                    biasGradient.Shape.GetDimension(1),
                    biasGradient.Shape.GetDimension(0));

                // bias
                this._context.CudnnContext.ConvolutionBackwardBias(1.0f, dOutputDesc, outputGradientStorage.DeviceBuffer, 0.0f,
                    dBiasDesc, biasGradientStorage.DeviceBuffer);
            }
        }

        public override void DoConvolution(Volume<float> filters, int pad, int stride, Volume<float> result)
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
                    cudnnConvolutionMode.CrossCorrelation, cudnnDataType.Float);

                dataDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    this.Shape.GetDimension(3),
                    this.Shape.GetDimension(2),
                    this.Shape.GetDimension(1),
                    this.Shape.GetDimension(0));

                filterDesc.SetFilter4dDescriptor(cudnnDataType.Float, cudnnTensorFormat.NCHW,
                    filters.Shape.GetDimension(3),
                    filters.Shape.GetDimension(2),
                    filters.Shape.GetDimension(1),
                    filters.Shape.GetDimension(0));

                outputDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
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

                this._context.CudnnContext.ConvolutionForward(1.0f,
                    dataDesc, inputStorage.DeviceBuffer,
                    filterDesc, filterStorage.DeviceBuffer,
                    convolutionDesc, algo, this._volumeStorage.ConvolutionStorage, 0.0f,
                    outputDesc, resultStorage.DeviceBuffer);
            }
        }

        public override void DoConvolutionGradient(Volume<float> filters, Volume<float> outputGradients,
            Volume<float> inputGradient, Volume<float> filterGradient, int pad,
            int stride)
        {
            var inputStorage = this._volumeStorage;
            var outputGradientStorage = outputGradients.Storage as VolumeStorage;
            var filterStorage = filters.Storage as VolumeStorage;
            var inputGradientStorage = inputGradient.Storage as VolumeStorage;
            var filterGradientStorage = filterGradient.Storage as VolumeStorage;

            // Copy to device if not already done
            inputStorage.CopyToDevice();
            outputGradientStorage.CopyToDevice();
            filterStorage.CopyToDevice();
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
                    cudnnConvolutionMode.CrossCorrelation, cudnnDataType.Float);

                dataDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    this.Shape.GetDimension(3),
                    this.Shape.GetDimension(2),
                    this.Shape.GetDimension(1),
                    this.Shape.GetDimension(0));

                dDataDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    this.Shape.GetDimension(3),
                    this.Shape.GetDimension(2),
                    this.Shape.GetDimension(1),
                    this.Shape.GetDimension(0));

                dOutputDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    outputGradients.Shape.GetDimension(3),
                    outputGradients.Shape.GetDimension(2),
                    outputGradients.Shape.GetDimension(1),
                    outputGradients.Shape.GetDimension(0));

                filterDesc.SetFilter4dDescriptor(cudnnDataType.Float, cudnnTensorFormat.NCHW,
                    filters.Shape.GetDimension(3),
                    filters.Shape.GetDimension(2),
                    filters.Shape.GetDimension(1),
                    filters.Shape.GetDimension(0));

                dfilterDesc.SetFilter4dDescriptor(cudnnDataType.Float, cudnnTensorFormat.NCHW,
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
                this._context.CudnnContext.ConvolutionBackwardFilter(1.0f, dataDesc, inputStorage.DeviceBuffer, dOutputDesc,
                    outputGradientStorage.DeviceBuffer, convolutionDesc, filterAlgo,
                    this._volumeStorage.ConvolutionBackwardFilterStorage, 0.0f, dfilterDesc,
                    filterGradientStorage.DeviceBuffer);

                // data
                if (this._volumeStorage.ConvolutionBackwardStorage == null || this._volumeStorage.ConvolutionBackwardStorage.Size != dataWorkspaceSize)
                {
                    this._volumeStorage.ConvolutionBackwardStorage = new CudaDeviceVariable<byte>(dataWorkspaceSize);
                }

                this._context.CudnnContext.ConvolutionBackwardData(1.0f,
                    filterDesc, filterStorage.DeviceBuffer,
                    dOutputDesc, outputGradientStorage.DeviceBuffer,
                    convolutionDesc, dataAlgo,
                    this._volumeStorage.ConvolutionBackwardStorage, 0.0f,
                    dDataDesc, inputGradientStorage.DeviceBuffer);
            }
        }

        public override void DoDivide(Volume<float> other, Volume<float> result)
        {
            _kernelLoader.RunKernel("Div", this, other, result);
        }

        public override void DoDropout(Volume<float> result, bool isTraining, float dropProbability)
        {
            DoDropout(result, dropProbability);
        }

        private void DoDropout(Volume<float> result, float dropProbability)
        {
            var resultStorage = result.Storage as VolumeStorage;
            if (resultStorage == null)
            {
                throw new ArgumentException($"{nameof(result)} storage should be VolumeStorage", nameof(result));
            }

            // Copy to device if not already done
            this._volumeStorage.CopyToDevice();
            resultStorage.CopyToDevice();

            using (var dropoutDesc = new DropoutDescriptor(this._context.CudnnContext))
            using (var srcDesc = new TensorDescriptor())
            using (var resultDesc = new TensorDescriptor())
            {
                srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    this.Shape.GetDimension(3), this.Shape.GetDimension(2),
                    this.Shape.GetDimension(1), this.Shape.GetDimension(0));

                resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    result.Shape.GetDimension(3), result.Shape.GetDimension(2),
                    result.Shape.GetDimension(1), result.Shape.GetDimension(0));

                var stateSize = this._context.CudnnContext.GetDropoutStateSize();
                if (this._volumeStorage.DropoutStateStorage == null || this._volumeStorage.DropoutStateStorage.Size != stateSize)
                {
                    this._volumeStorage.DropoutStateStorage = new CudaDeviceVariable<byte>(stateSize);
                }

                dropoutDesc.SetDropoutDescriptor(dropProbability, this._volumeStorage.DropoutStateStorage, stateSize, 0);

                var reserveSpace = this._context.CudnnContext.GetDropoutReserveSpaceSize(srcDesc);
                reserveSpace = reserveSpace == 0 ? new SizeT(1) : reserveSpace;

                if (this._volumeStorage.DropoutStorage == null || this._volumeStorage.DropoutStorage.Size != reserveSpace)
                {
                    this._volumeStorage.DropoutStorage = new CudaDeviceVariable<byte>(reserveSpace);
                }

                this._context.CudnnContext.DropoutForward(dropoutDesc, srcDesc, this._volumeStorage.DeviceBuffer, resultDesc, resultStorage.DeviceBuffer, this._volumeStorage.DropoutStorage);
            }
        }

        public override void DoDropoutGradient(Volume<float> input, Volume<float> outputGradient, Volume<float> inputGradient, float dropProbability)
        {
            var inputStorage = this._volumeStorage;
            var outputGradientStorage = outputGradient.Storage as VolumeStorage;
            var inputGradientStorage = inputGradient.Storage as VolumeStorage;

            // Copy to device if not already done
            inputStorage.CopyToDevice();
            outputGradientStorage.CopyToDevice();
            inputGradientStorage.CopyToDevice();

            using (var dropoutDesc = new DropoutDescriptor(this._context.CudnnContext))
            using (var dOutputDesc = new TensorDescriptor())
            using (var dDataDesc = new TensorDescriptor())
            {
                var stateSize = this._context.CudnnContext.GetDropoutStateSize();
                if (this._volumeStorage.DropoutStateStorage == null || this._volumeStorage.DropoutStateStorage.Size != stateSize)
                {
                    this._volumeStorage.DropoutStateStorage = new CudaDeviceVariable<byte>(stateSize);
                }

                dropoutDesc.SetDropoutDescriptor(dropProbability, this._volumeStorage.DropoutStateStorage, stateSize, 0);

                dDataDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    this.Shape.GetDimension(3),
                    this.Shape.GetDimension(2),
                    this.Shape.GetDimension(1),
                    this.Shape.GetDimension(0));

                dOutputDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    outputGradient.Shape.GetDimension(3),
                    outputGradient.Shape.GetDimension(2),
                    outputGradient.Shape.GetDimension(1),
                    outputGradient.Shape.GetDimension(0));

                this._context.CudnnContext.DropoutBackward(dropoutDesc,
                    dOutputDesc, outputGradientStorage.DeviceBuffer,
                    dDataDesc, inputGradientStorage.DeviceBuffer,
                    this._volumeStorage.DropoutStorage);
            }
        }

        public override void DoExp(Volume<float> result)
        {
            _kernelLoader.RunKernel("Exp", this, result);
        }

        public override void DoLeakyRelu(Volume<float> result)
        {
            _kernelLoader.RunKernel("LeakyRelu", this, result);
        }

        public override void DoLeakyReluGradient(Volume<float> input, Volume<float> outputGradient, Volume<float> inputGradient)
        {
            _kernelLoader.RunKernel("LeakyReluGradient", this, outputGradient, inputGradient);
        }

        public override void DoLog(Volume<float> result)
        {
            _kernelLoader.RunKernel("Log", this, result);
        }

        public override void DoMax(Volume<float> result)
        {
            DoReduce(result, cudnnReduceTensorOp.Max);
        }

        public override void DoMin(Volume<float> result)
        {
            DoReduce(result, cudnnReduceTensorOp.Min);
        }

        public override void DoMultiply(Volume<float> right, Volume<float> result)
        {
            var resultStorage = result.Storage as VolumeStorage;
            if (resultStorage == null)
            {
                throw new ArgumentException($"{nameof(result)} storage should be VolumeStorage", nameof(result));
            }

            var rightStorage = right.Storage as VolumeStorage;
            if (rightStorage == null)
            {
                throw new ArgumentException($"{nameof(right)} storage should be VolumeStorage", nameof(right));
            }

            // Copy to device if not already done
            this._volumeStorage.CopyToDevice();
            rightStorage.CopyToDevice();
            resultStorage.CopyToDevice();

            var aStorage = this._volumeStorage;
            var bStorage = rightStorage;
            if (bStorage.Shape.TotalLength > aStorage.Shape.TotalLength)
            {
                aStorage = rightStorage;
                bStorage = this._volumeStorage;
            }
            var bShape = bStorage.Shape;

            var n = aStorage.Shape.GetDimension(3);
            var c = aStorage.Shape.GetDimension(2);
            var h = aStorage.Shape.GetDimension(1);
            var w = aStorage.Shape.GetDimension(0);

            // Add tensors
            using (var descA = new TensorDescriptor())
            using (var descB = new TensorDescriptor())
            using (var descC = new TensorDescriptor())
            {
                descA.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
                descB.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, bShape.GetDimension(3), bShape.GetDimension(2), bShape.GetDimension(1),
                    bShape.GetDimension(0));
                descC.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);

                using (var opt = new OpTensorDescriptor(this._context.CudnnContext))
                {
                    opt.SetOpTensorDescriptor(
                        cudnnOpTensorOp.OpTensorMul,
                        cudnnDataType.Float,
                        cudnnNanPropagation.PropagateNan);

                    var one = 1.0f;
                    var zero = 0.0f;

                    var status = CudaDNNNativeMethods.cudnnOpTensor(
                        this._context.CudnnContext.Handle,
                        opt.Desc,
                        ref one, descA.Desc, aStorage.DeviceBuffer.DevicePointer,
                        ref one, descB.Desc, bStorage.DeviceBuffer.DevicePointer,
                        ref zero, descC.Desc, resultStorage.DeviceBuffer.DevicePointer);

                    if (status != cudnnStatus.Success)
                    {
                        throw new Exception(CudaDNNNativeMethods.cudnnGetErrorString(status));
                    }

                    resultStorage.Location = DataLocation.Device;
                }
            }
        }

        public override void DoMultiply(Volume<float> result, float factor)
        {
            var resultStorage = result.Storage as VolumeStorage;
            if (resultStorage == null)
            {
                throw new ArgumentException($"{nameof(result)} storage should be VolumeStorage", nameof(result));
            }

            // Copy to device if not already done
            this._volumeStorage.CopyToDevice();
            resultStorage.CopyFrom(this._volumeStorage);

            // Add tensors
            using (var resultDesc = new TensorDescriptor())
            {
                var n = result.Shape.GetDimension(3);
                var c = result.Shape.GetDimension(2);
                var h = result.Shape.GetDimension(1);
                var w = result.Shape.GetDimension(0);

                resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);

                this._context.CudnnContext.ScaleTensor(resultDesc, resultStorage.DeviceBuffer, factor);
            }
        }

        public override void DoNegate(Volume<float> result)
        {
            DoMultiply(result, -1.0f);
        }

        public override void DoNorm1(Volume<float> result)
        {
            DoReduce(result, cudnnReduceTensorOp.Norm1);
        }

        public override void DoPool(Volume<float> result, int windowWidth, int windowHeight,
            int horizontalPad, int verticalPad, int horizontalStride, int verticalStride)
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
            using (var poolingDesc = new PoolingDescriptor())
            using (var srcDesc = new TensorDescriptor())
            using (var resultDesc = new TensorDescriptor())
            {
                srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    this.Shape.GetDimension(3), this.Shape.GetDimension(2),
                    this.Shape.GetDimension(1), this.Shape.GetDimension(0));

                resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    result.Shape.GetDimension(3), result.Shape.GetDimension(2),
                    result.Shape.GetDimension(1), result.Shape.GetDimension(0));

                poolingDesc.SetPooling2dDescriptor(cudnnPoolingMode.Max, cudnnNanPropagation.NotPropagateNan,
                    windowHeight, windowWidth,
                    verticalPad, horizontalPad, verticalStride, horizontalStride);

                this._context.CudnnContext.PoolingForward(poolingDesc, 1.0f, srcDesc, this._volumeStorage.DeviceBuffer, 0.0f,
                    resultDesc, resultStorage.DeviceBuffer);
            }
        }

        public override void DoPoolGradient(Volume<float> input, Volume<float> outputGradient,
            Volume<float> inputGradient, int windowWidth, int windowHeight,
            int horizontalPad, int verticalPad, int horizontalStride, int verticalStride)
        {
            var inputStorage = input.Storage as VolumeStorage;
            var inputGradientStorage = inputGradient.Storage as VolumeStorage;
            var outputStorage = this._volumeStorage;
            var outputGradientStorage = outputGradient.Storage as VolumeStorage;

            // Copy to device if not already done
            //outputStorage.CopyToDevice();
            outputGradientStorage.CopyToDevice();
            inputStorage.CopyToDevice();
            inputGradientStorage.CopyToDevice();

            // Synchro
            this._context.DefaultStream.Synchronize();

            using (var poolingDesc = new PoolingDescriptor())
            using (var srcDesc = new TensorDescriptor())
            using (var srcDiffDesc = new TensorDescriptor())
            using (var destDesc = new TensorDescriptor())
            using (var destDiffDesc = new TensorDescriptor())
            {
                var n = this.Shape.GetDimension(3);
                var c = this.Shape.GetDimension(2);
                var h = this.Shape.GetDimension(1);
                var w = this.Shape.GetDimension(0);

                srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
                srcDiffDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);

                destDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    inputStorage.Shape.GetDimension(3), inputStorage.Shape.GetDimension(2),
                    inputStorage.Shape.GetDimension(1), inputStorage.Shape.GetDimension(0));
                destDiffDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    inputStorage.Shape.GetDimension(3), inputStorage.Shape.GetDimension(2),
                    inputStorage.Shape.GetDimension(1), inputStorage.Shape.GetDimension(0));

                poolingDesc.SetPooling2dDescriptor(cudnnPoolingMode.Max, cudnnNanPropagation.NotPropagateNan,
                    windowHeight, windowWidth,
                    verticalPad, horizontalPad, verticalStride, horizontalStride);

                this._context.CudnnContext.PoolingBackward(poolingDesc, 1.0f,
                    srcDesc, outputStorage.DeviceBuffer,
                    srcDiffDesc, outputGradientStorage.DeviceBuffer,
                    destDesc, inputStorage.DeviceBuffer,
                    0.0f,
                    destDiffDesc, inputGradientStorage.DeviceBuffer);
            }
        }

        public override void DoReduce(Volume<float> result, TensorReduceOp op)
        {
            DoReduce(result, op.ToCudnn());
        }

        private void DoReduce(Volume<float> result, cudnnReduceTensorOp op)
        {
            if (this.Shape.Equals(result.Shape))
            {
                result.Storage.CopyFrom(this.Storage);
                return;
            }

            var aStorage = this._volumeStorage;
            var cStorage = result.Storage as VolumeStorage;

            // Copy to device if not already done
            aStorage.CopyToDevice();
            cStorage.CopyToDevice();

            using (var reduceTensorDesc = new ReduceTensorDescriptor())
            using (var aDesc = new TensorDescriptor())
            using (var cDesc = new TensorDescriptor())
            {
                var an = this.Shape.GetDimension(3);
                var ac = this.Shape.GetDimension(2);
                var ah = this.Shape.GetDimension(1);
                var aw = this.Shape.GetDimension(0);

                var cn = result.Shape.GetDimension(3);
                var cc = result.Shape.GetDimension(2);
                var ch = result.Shape.GetDimension(1);
                var cw = result.Shape.GetDimension(0);

                aDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, an, ac, ah, aw);
                cDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, cn, cc, ch, cw);

                reduceTensorDesc.SetReduceTensorDescriptor(op, cudnnDataType.Float, cudnnNanPropagation.NotPropagateNan, cudnnReduceTensorIndices.NoIndices,
                    cudnnIndicesType.Indices32Bit);

                var workspaceSize = this._context.CudnnContext.GetReductionWorkspaceSize(reduceTensorDesc, aDesc, cDesc);
                workspaceSize = workspaceSize == 0 ? new SizeT(1) : workspaceSize;

                if (this._volumeStorage.ReductionStorage == null || this._volumeStorage.ReductionStorage.Size != workspaceSize)
                {
                    this._volumeStorage.ReductionStorage = new CudaDeviceVariable<byte>(workspaceSize);
                }

                this._context.CudnnContext.ReduceTensor(reduceTensorDesc,
                    CudaDeviceVariable<uint>.Null,
                    this._volumeStorage.ReductionStorage,
                    this._volumeStorage.ReductionStorage.SizeInBytes,
                    1.0f, aDesc, aStorage.DeviceBuffer,
                    0.0f, cDesc, cStorage.DeviceBuffer);
            }
        }

        public override void DoRelu(Volume<float> result)
        {
            DoActivation(result, cudnnActivationMode.Relu);
        }

        public override void DoReluGradient(Volume<float> input, Volume<float> outputGradient,
            Volume<float> inputGradient)
        {
            DoActivationGradient(input, outputGradient, inputGradient, cudnnActivationMode.Relu);
        }

        public override void DoSigmoid(Volume<float> result)
        {
            DoActivation(result, cudnnActivationMode.Sigmoid);
        }

        public override void DoSigmoidGradient(Volume<float> input, Volume<float> outputGradient,
            Volume<float> inputGradient)
        {
            DoActivationGradient(input, outputGradient, inputGradient, cudnnActivationMode.Sigmoid);
        }

        public override void DoSoftmax(Volume<float> output)
        {
            var inputStorage = this._volumeStorage;
            var outputStorage = output.Storage as VolumeStorage;

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

                srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
                destDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);

                this._context.CudnnContext.SoftmaxForward(cudnnSoftmaxAlgorithm.Accurate, cudnnSoftmaxMode.Channel, 1.0f,
                    srcDesc, inputStorage.DeviceBuffer, 0.0f,
                    destDesc, outputStorage.DeviceBuffer);
            }
        }

        public override void DoSoftmaxGradient(Volume<float> outputGradient, Volume<float> inputGradient)
        {
            var inputGradientStorage = inputGradient.Storage as VolumeStorage;
            var outputGradientStorage = outputGradient.Storage as VolumeStorage;
            var outputStorage = this._volumeStorage;

            // Copy to device if not already done
            outputStorage.CopyToDevice();
            outputGradientStorage.CopyToDevice();
            inputGradientStorage.CopyToDevice();

            // Synchro
            this._context.DefaultStream.Synchronize();

            using (var srcDesc = new TensorDescriptor())
            using (var srcDiffDesc = new TensorDescriptor())
            using (var destDiffDesc = new TensorDescriptor())
            {
                var n = this.Shape.GetDimension(3);
                var c = this.Shape.GetDimension(2);
                var h = this.Shape.GetDimension(1);
                var w = this.Shape.GetDimension(0);

                srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
                srcDiffDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
                destDiffDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);

                this._context.CudnnContext.SoftmaxBackward(cudnnSoftmaxAlgorithm.Accurate, cudnnSoftmaxMode.Channel, 1.0f,
                    srcDesc, outputStorage.DeviceBuffer,
                    srcDiffDesc, outputGradientStorage.DeviceBuffer,
                    0.0f,
                    destDiffDesc, inputGradientStorage.DeviceBuffer);
            }
        }

        public override void DoSubtractFrom(Volume<float> other, Volume<float> result)
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

            resultStorage.CopyFrom(otherStorage);
            this._volumeStorage.CopyToDevice();

            // Add tensors
            using (var subtractorDesc = new TensorDescriptor())
            using (var resultDesc = new TensorDescriptor())
            {
                subtractorDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    other.Shape.GetDimension(3),
                    other.Shape.GetDimension(2),
                    other.Shape.GetDimension(1),
                    other.Shape.GetDimension(0));

                resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    this.Shape.GetDimension(3),
                    this.Shape.GetDimension(2),
                    this.Shape.GetDimension(1),
                    this.Shape.GetDimension(0));

                this._context.CudnnContext.AddTensor(
                    -1.0f, subtractorDesc, this._volumeStorage.DeviceBuffer,
                    1.0f, resultDesc, resultStorage.DeviceBuffer);
            }
        }

        public override void DoSum(Volume<float> result)
        {
            DoReduce(result, TensorReduceOp.Add);
        }

        public override void DoTanh(Volume<float> result)
        {
            DoActivation(result, cudnnActivationMode.Tanh);
        }

        public override void DoTanhGradient(Volume<float> input, Volume<float> outputGradient,
            Volume<float> inputGradient)
        {
            DoActivationGradient(input, outputGradient, inputGradient, cudnnActivationMode.Tanh);
        }

        private void LoadKernels()
        {
            if (_kernelLoader == null)
            {
                _kernelLoader = new KernelLoader<float>(this._context);

                var assembly = Assembly.GetExecutingAssembly();
                using (Stream stream = assembly.GetManifestResourceStream("ConvNetSharp.Volume.GPU.Single.Kernels.log.cu"))
                {
                    _kernelLoader.LoadKernel("Log", stream);
                }

                using (Stream stream = assembly.GetManifestResourceStream("ConvNetSharp.Volume.GPU.Single.Kernels.exp.cu"))
                {
                    _kernelLoader.LoadKernel("Exp", stream);
                }

                using (Stream stream = assembly.GetManifestResourceStream("ConvNetSharp.Volume.GPU.Single.Kernels.div.cu"))
                {
                    _kernelLoader.LoadKernel("Div", stream);
                }

                using (Stream stream = assembly.GetManifestResourceStream("ConvNetSharp.Volume.GPU.Single.Kernels.leakyrelu.cu"))
                {
                    _kernelLoader.LoadKernel("LeakyRelu", stream);
                }

                using (Stream stream = assembly.GetManifestResourceStream("ConvNetSharp.Volume.GPU.Single.Kernels.leakyrelu_gradient.cu"))
                {
                    _kernelLoader.LoadKernel("LeakyReluGradient", stream);
                }
            }
        }
    }
}