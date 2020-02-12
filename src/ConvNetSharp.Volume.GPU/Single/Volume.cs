using System;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using ManagedCuda.CudaDNN;
using Math = System.Math;

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

            this.LoadKernels();
        }

        internal Volume(float[] array, Shape shape) : base(new VolumeStorage(array, shape, GpuContext.Default))
        {
            this._context = GpuContext.Default;
            this._volumeStorage = this.Storage as VolumeStorage;

            this.LoadKernels();
        }

        public Volume(float[] array, Shape shape, GpuContext context) : base(new VolumeStorage(array, shape, context))
        {
            this._context = context;
            this._volumeStorage = this.Storage as VolumeStorage;

            this.LoadKernels();
        }

        private void Activation(Volume<float> result, cudnnActivationMode mode)
        {
            if (!(result.Storage is VolumeStorage resultStorage))
            {
                throw new ArgumentException($"{nameof(result)} storage should be VolumeStorage", nameof(result));
            }

            // Copy to device if not already done
            this._volumeStorage.CopyToDevice();
            resultStorage.CopyToDevice();

            // Synchro
            this._context.DefaultStream.Synchronize();

            // Relu
            using var activationDesc = new ActivationDescriptor();
            using var srcDesc = new TensorDescriptor();
            using var resultDesc = new TensorDescriptor();

            var n = result.Shape.Dimensions[3];
            var c = result.Shape.Dimensions[2];
            var h = result.Shape.Dimensions[1];
            var w = result.Shape.Dimensions[0];

            srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
            resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
            activationDesc.SetActivationDescriptor(mode, cudnnNanPropagation.NotPropagateNan, 0.0);

            this._context.CudnnContext.ActivationForward(activationDesc,
                1.0f, srcDesc, this._volumeStorage.DeviceBuffer,
                0.0f, resultDesc, resultStorage.DeviceBuffer);
        }

        public override void Activation(ActivationType type, Volume<float> result)
        {
            this.Activation(result, type.ToCudnn());
        }

        private void ActivationGradient(Volume<float> input, Volume<float> outputGradient, Volume<float> inputGradient, cudnnActivationMode mode)
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

            using var activationDesc = new ActivationDescriptor();
            using var srcDesc = new TensorDescriptor();
            using var srcDiffDesc = new TensorDescriptor();
            using var destDesc = new TensorDescriptor();
            using var destDiffDesc = new TensorDescriptor();

            var n = this.Shape.Dimensions[3];
            var c = this.Shape.Dimensions[2];
            var h = this.Shape.Dimensions[1];
            var w = this.Shape.Dimensions[0];

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

        public override void ActivationGradient(Volume<float> input, Volume<float> outputGradient, ActivationType type, Volume<float> result)
        {
            this.ActivationGradient(input, outputGradient, result, type.ToCudnn());
        }

        public override void Add(Volume<float> result)
        {
            var inputStorage = this.Storage as VolumeStorage;
            var resultStorage = result.Storage as VolumeStorage;

            if (inputStorage == null)
            {
                throw new ArgumentException($"{nameof(result)} storage should be VolumeStorage", nameof(result));
            }

            if (resultStorage == null)
            {
                throw new ArgumentException($"{nameof(result)} storage should be VolumeStorage", nameof(result));
            }

            var resultDim3 = result.Shape.Dimensions[3];
            var resultDim2 = result.Shape.Dimensions[2];
            var resultDim1 = result.Shape.Dimensions[1];
            var resultDim0 = result.Shape.Dimensions[0];

            var dim3 = this.Shape.Dimensions[3];
            var dim2 = this.Shape.Dimensions[2];
            var dim1 = this.Shape.Dimensions[1];
            var dim0 = this.Shape.Dimensions[0];

            if (dim0 == 1 && dim1 == 1 && dim2 == 1)
            {
                resultDim3 = (int)result.Shape.TotalLength;
                resultDim0 = 1;
                resultDim1 = 1;
                resultDim2 = 1;
            }

            // Copy to device if not already done
            inputStorage.CopyToDevice();
            resultStorage.CopyToDevice();

            // Add tensors
            using var otherDesc = new TensorDescriptor();
            using var resultDesc = new TensorDescriptor();

            resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                resultDim3,
                resultDim2,
                resultDim1,
                resultDim0);

            otherDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                dim3,
                dim2,
                dim1,
                dim0);

            this._context.CudnnContext.AddTensor(
                1.0f, otherDesc, inputStorage.DeviceBuffer,
                1.0f, resultDesc, resultStorage.DeviceBuffer);
        }

        public override void Add(Volume<float> other, Volume<float> result)
        {
            if (this != result)
            {
                result.Clear();
                this.Add(result);
            }

            other.Add(result);
        }

        public override void BiasGradient(Volume<float> biasGradient)
        {
            var outputGradientStorage = this._volumeStorage;
            var biasGradientStorage = biasGradient.Storage as VolumeStorage;

            // Copy to device if not already done
            outputGradientStorage.CopyToDevice();
            biasGradientStorage.CopyToDevice();

            using var dOutputDesc = new TensorDescriptor();
            using var dBiasDesc = new TensorDescriptor();

            dOutputDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                this.Shape.Dimensions[3],
                this.Shape.Dimensions[2],
                this.Shape.Dimensions[1],
                this.Shape.Dimensions[0]);

            dBiasDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                biasGradient.Shape.Dimensions[3],
                biasGradient.Shape.Dimensions[2],
                biasGradient.Shape.Dimensions[1],
                biasGradient.Shape.Dimensions[0]);

            // bias
            this._context.CudnnContext.ConvolutionBackwardBias(1.0f, dOutputDesc, outputGradientStorage.DeviceBuffer, 0.0f,
                dBiasDesc, biasGradientStorage.DeviceBuffer);
        }

        public override void Concat(Volume<float> right, Volume<float> result)
        {
            var batchSize = Math.Max(this.Shape.Dimensions[3], right.Shape.Dimensions[3]);
            var elementPerBatch = result.Shape.TotalLength / batchSize;

            // mode 0: none of the inputs are scalars
            // mode 1: left is a scalar
            // mode 2: right is a scalar
            var mode = this.Shape.TotalLength > 1 && right.Shape.TotalLength > 1 ? 0 : this.Shape.TotalLength == 1 && right.Shape.TotalLength > 1 ? 1 : 2;
            var threshold = mode == 1 ? 1 : this.Shape.TotalLength / batchSize;

            _kernelLoader.RunKernel("concat", this, right, result, elementPerBatch, threshold, mode);
        }

        public override void Convolution(Volume<float> filters, int pad, int stride, Volume<float> result)
        {
            if (!(result.Storage is VolumeStorage resultStorage))
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

            using var dataDesc = new TensorDescriptor();
            using var filterDesc = new FilterDescriptor();
            using var outputDesc = new TensorDescriptor();
            using var convolutionDesc = new ConvolutionDescriptor();

            convolutionDesc.SetConvolution2dDescriptor(pad, pad, stride, stride, 1, 1,
                cudnnConvolutionMode.CrossCorrelation, cudnnDataType.Float);

            dataDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                this.Shape.Dimensions[3],
                this.Shape.Dimensions[2],
                this.Shape.Dimensions[1],
                this.Shape.Dimensions[0]);

            filterDesc.SetFilter4dDescriptor(cudnnDataType.Float, cudnnTensorFormat.NCHW,
                filters.Shape.Dimensions[3],
                filters.Shape.Dimensions[2],
                filters.Shape.Dimensions[1],
                filters.Shape.Dimensions[0]);

            outputDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                result.Shape.Dimensions[3],
                result.Shape.Dimensions[2],
                result.Shape.Dimensions[1],
                result.Shape.Dimensions[0]);

            var algo = this._context.CudnnContext.GetConvolutionForwardAlgorithm(
                dataDesc, filterDesc,
                convolutionDesc, outputDesc,
                cudnnConvolutionFwdPreference.PreferFastest, IntPtr.Zero);

            var workspaceSize = this._context.CudnnContext.GetConvolutionForwardWorkspaceSize(
                dataDesc, filterDesc,
                convolutionDesc, outputDesc, algo);
            workspaceSize = workspaceSize == 0 ? new SizeT(1) : workspaceSize;

            if (resultStorage.ConvolutionStorage == null || resultStorage.ConvolutionStorage.Size != workspaceSize)
            {
                Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "DoConvolution", workspaceSize);
                if (resultStorage.ConvolutionStorage != null)
                {
                    Debug.WriteLine("{0:G}, {1}: \t {2} -> {3}", DateTime.Now, "DoConvolution", resultStorage.ConvolutionStorage.Size, workspaceSize);
                }

                resultStorage.ConvolutionStorage = new CudaDeviceVariable<byte>(workspaceSize);
            }

            this._context.CudnnContext.ConvolutionForward(1.0f,
                dataDesc, inputStorage.DeviceBuffer,
                filterDesc, filterStorage.DeviceBuffer,
                convolutionDesc, algo, resultStorage.ConvolutionStorage, 0.0f,
                outputDesc, resultStorage.DeviceBuffer);
        }

        public override void ConvolutionGradient(Volume<float> filters, Volume<float> outputGradients,
            Volume<float> filterGradient, int pad, int stride, Volume<float> inputGradient)
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

            using var dataDesc = new TensorDescriptor();
            using var filterDesc = new FilterDescriptor();
            using var dDataDesc = new TensorDescriptor();
            using var dOutputDesc = new TensorDescriptor();
            using var dfilterDesc = new FilterDescriptor();
            using var convolutionDesc = new ConvolutionDescriptor();

            convolutionDesc.SetConvolution2dDescriptor(pad, pad, stride, stride, 1, 1,
                cudnnConvolutionMode.CrossCorrelation, cudnnDataType.Float);

            dataDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                this.Shape.Dimensions[3],
                this.Shape.Dimensions[2],
                this.Shape.Dimensions[1],
                this.Shape.Dimensions[0]);

            dDataDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                this.Shape.Dimensions[3],
                this.Shape.Dimensions[2],
                this.Shape.Dimensions[1],
                this.Shape.Dimensions[0]);

            dOutputDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                outputGradients.Shape.Dimensions[3],
                outputGradients.Shape.Dimensions[2],
                outputGradients.Shape.Dimensions[1],
                outputGradients.Shape.Dimensions[0]);

            filterDesc.SetFilter4dDescriptor(cudnnDataType.Float, cudnnTensorFormat.NCHW,
                filters.Shape.Dimensions[3],
                filters.Shape.Dimensions[2],
                filters.Shape.Dimensions[1],
                filters.Shape.Dimensions[0]);

            dfilterDesc.SetFilter4dDescriptor(cudnnDataType.Float, cudnnTensorFormat.NCHW,
                filters.Shape.Dimensions[3],
                filters.Shape.Dimensions[2],
                filters.Shape.Dimensions[1],
                filters.Shape.Dimensions[0]);

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
            if (inputGradientStorage.ConvolutionBackwardFilterStorage == null || inputGradientStorage.ConvolutionBackwardFilterStorage.Size != filterWorkspaceSize)
            {
                Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "DoConvolutionGradient - filter", filterWorkspaceSize);
                inputGradientStorage.ConvolutionBackwardFilterStorage = new CudaDeviceVariable<byte>(filterWorkspaceSize);
            }

            this._context.CudnnContext.ConvolutionBackwardFilter(1.0f, dataDesc, inputStorage.DeviceBuffer, dOutputDesc,
                outputGradientStorage.DeviceBuffer, convolutionDesc, filterAlgo,
                inputGradientStorage.ConvolutionBackwardFilterStorage, 0.0f, dfilterDesc,
                filterGradientStorage.DeviceBuffer);

            // data
            if (inputGradientStorage.ConvolutionBackwardStorage == null || inputGradientStorage.ConvolutionBackwardStorage.Size != dataWorkspaceSize)
            {
                Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "DoConvolutionGradient - workspace", dataWorkspaceSize);
                inputGradientStorage.ConvolutionBackwardStorage = new CudaDeviceVariable<byte>(dataWorkspaceSize);
            }

            this._context.CudnnContext.ConvolutionBackwardData(1.0f,
                filterDesc, filterStorage.DeviceBuffer,
                dOutputDesc, outputGradientStorage.DeviceBuffer,
                convolutionDesc, dataAlgo,
                inputGradientStorage.ConvolutionBackwardStorage, 0.0f,
                dDataDesc, inputGradientStorage.DeviceBuffer);
        }

        public override void Divide(Volume<float> other, Volume<float> result)
        {
            _kernelLoader.RunKernel("div", this, other, result, other.Shape.TotalLength == 1 ? 1 : 0);
        }

        public override void Dropout(float dropProbability, Volume<float> result)
        {
            if (!(result.Storage is VolumeStorage resultStorage))
            {
                throw new ArgumentException($"{nameof(result)} storage should be VolumeStorage", nameof(result));
            }

            // Copy to device if not already done
            this._volumeStorage.CopyToDevice();
            resultStorage.CopyToDevice();

            using var dropoutDesc = new DropoutDescriptor(this._context.CudnnContext);
            using var srcDesc = new TensorDescriptor();
            using var resultDesc = new TensorDescriptor();

            srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                this.Shape.Dimensions[3],
                this.Shape.Dimensions[2],
                this.Shape.Dimensions[1],
                this.Shape.Dimensions[0]);

            resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                result.Shape.Dimensions[3],
                result.Shape.Dimensions[2],
                result.Shape.Dimensions[1],
                result.Shape.Dimensions[0]);

            var stateSize = this._context.CudnnContext.GetDropoutStateSize();
            if (resultStorage.DropoutStateStorage == null || resultStorage.DropoutStateStorage.Size != stateSize)
            {
                Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "DoDropout - state", stateSize);
                resultStorage.DropoutStateStorage = new CudaDeviceVariable<byte>(stateSize);
            }

            dropoutDesc.SetDropoutDescriptor(dropProbability, resultStorage.DropoutStateStorage, stateSize, 0);

            var reserveSpace = this._context.CudnnContext.GetDropoutReserveSpaceSize(srcDesc);
            reserveSpace = reserveSpace == 0 ? new SizeT(1) : reserveSpace;

            if (resultStorage.DropoutStorage == null || resultStorage.DropoutStorage.Size != reserveSpace)
            {
                Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "DoDropout", reserveSpace);
                resultStorage.DropoutStorage = new CudaDeviceVariable<byte>(reserveSpace);
            }

            this._context.CudnnContext.DropoutForward(dropoutDesc, srcDesc, this._volumeStorage.DeviceBuffer, resultDesc, resultStorage.DeviceBuffer,
                resultStorage.DropoutStorage);
        }

        public override void DropoutGradient(Volume<float> input, Volume<float> outputGradient, float dropProbability, Volume<float> inputGradient)
        {
            var outputStorage = this.Storage as VolumeStorage;
            var inputStorage = input.Storage as VolumeStorage;
            var outputGradientStorage = outputGradient.Storage as VolumeStorage;
            var inputGradientStorage = inputGradient.Storage as VolumeStorage;

            // Copy to device if not already done
            inputStorage.CopyToDevice();
            outputGradientStorage.CopyToDevice();
            inputGradientStorage.CopyToDevice();

            using var dropoutDesc = new DropoutDescriptor(this._context.CudnnContext);
            using var dOutputDesc = new TensorDescriptor();
            using var dDataDesc = new TensorDescriptor();

            var stateSize = this._context.CudnnContext.GetDropoutStateSize();
            if (outputStorage.DropoutStateStorage == null || outputStorage.DropoutStateStorage.Size != stateSize)
            {
                Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "DoDropoutGradient", stateSize);
                outputStorage.DropoutStateStorage = new CudaDeviceVariable<byte>(stateSize);
            }

            dropoutDesc.SetDropoutDescriptor(dropProbability, outputStorage.DropoutStateStorage, stateSize, 0);

            dDataDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                this.Shape.Dimensions[3],
                this.Shape.Dimensions[2],
                this.Shape.Dimensions[1],
                this.Shape.Dimensions[0]);

            dOutputDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                outputGradient.Shape.Dimensions[3],
                outputGradient.Shape.Dimensions[2],
                outputGradient.Shape.Dimensions[1],
                outputGradient.Shape.Dimensions[0]);

            this._context.CudnnContext.DropoutBackward(dropoutDesc,
                dOutputDesc, outputGradientStorage.DeviceBuffer,
                dDataDesc, inputGradientStorage.DeviceBuffer,
                outputStorage.DropoutStorage);
        }

        public override void Exp(Volume<float> result)
        {
            _kernelLoader.RunKernel("exp", this, result);
        }

        public override void Extract(int length, int offset, Volume<float> result)
        {
            _kernelLoader.RunKernel("extract", this, result, new object[] { length, offset, this.Shape.TotalLength });
        }

        public override void LeakyRelu(float alpha, Volume<float> result)
        {
            _kernelLoader.RunKernel("leakyrelu", this, result, new object[] { alpha });
        }

        public override void LeakyReluGradient(Volume<float> outputGradient, Volume<float> inputGradient, float alpha)
        {
            _kernelLoader.RunKernel("leakyrelu_gradient", this, outputGradient, inputGradient, alpha);
        }

        private void LoadKernels()
        {
            if (_kernelLoader != null)
            {
                return;
            }

            _kernelLoader = new KernelLoader<float>(this._context);

            var assembly = Assembly.GetExecutingAssembly();

            // Retrieve all kernels from resources
            var regex = new Regex(@"ConvNetSharp\.Volume\.GPU\.Single\.Kernels\.(.*)\.cu", RegexOptions.Compiled);
            var tuples = assembly.GetManifestResourceNames().Where(o => regex.IsMatch(o)).Select(o => new Tuple<string, string>(o, regex.Match(o).Groups[1].Value));

            foreach (var t in tuples)
            {
                using var stream = assembly.GetManifestResourceStream(t.Item1);

                _kernelLoader.LoadKernel(t.Item2, stream);
            }
        }

        public override void Log(Volume<float> result)
        {
            _kernelLoader.RunKernel("log", this, result);
        }

        public override void MatMultiply(Volume<float> right, Volume<float> result)
        {
            var leftStorage = this.Storage as VolumeStorage;
            var rightStorage = right.Storage as VolumeStorage;
            var resultStorage = result.Storage as VolumeStorage;
            leftStorage.CopyToDevice();
            rightStorage.CopyToDevice();
            resultStorage.CopyToDevice();

            // https://www.christophlassner.de/using-blas-from-c-with-row-major-data.html

            var m = this.Shape.Dimensions[1];
            var n = right.Shape.Dimensions[0];
            var k = this.Shape.Dimensions[0];

            var broadCastLeft = this.Shape.Dimensions[3] == 1;
            var broadCastRight = right.Shape.Dimensions[3] == 1;

            for (var b = 0; b < result.Shape.Dimensions[3]; b++) // for each batch
            {
                this._context.CudaBlasHandle.Gemm(Operation.NonTranspose, Operation.NonTranspose,
                    n, m, k,
                    1.0f,
                    broadCastRight ? rightStorage.DeviceBuffer : new CudaDeviceVariable<float>(rightStorage.DeviceBuffer.DevicePointer + b * k * n * sizeof(float)), n,
                    broadCastLeft ? leftStorage.DeviceBuffer : new CudaDeviceVariable<float>(leftStorage.DeviceBuffer.DevicePointer + b * k * m * sizeof(float)), k,
                    0.0f,
                    new CudaDeviceVariable<float>(resultStorage.DeviceBuffer.DevicePointer + b * n * m * sizeof(float)), n);
            }
        }

        public override void Max(Volume<float> result)
        {
            this.Reduce(result, cudnnReduceTensorOp.Max);
        }

        public override void Min(Volume<float> result)
        {
            this.Reduce(result, cudnnReduceTensorOp.Min);
        }

        public override void Multiply(Volume<float> right, Volume<float> result)
        {
            this.Op(right, cudnnOpTensorOp.OpTensorMul, result);
        }

        public override void Multiply(float factor, Volume<float> result)
        {
            if (!(result.Storage is VolumeStorage resultStorage))
            {
                throw new ArgumentException($"{nameof(result)} storage should be VolumeStorage", nameof(result));
            }

            // Copy to device if not already done
            this._volumeStorage.CopyToDevice();
            resultStorage.CopyFrom(this._volumeStorage);

            // Add tensors
            using var resultDesc = new TensorDescriptor();

            var n = result.Shape.Dimensions[3];
            var c = result.Shape.Dimensions[2];
            var h = result.Shape.Dimensions[1];
            var w = result.Shape.Dimensions[0];

            resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);

            this._context.CudnnContext.ScaleTensor(resultDesc, resultStorage.DeviceBuffer, factor);
        }

        public override void Negate(Volume<float> result)
        {
            this.Multiply(-1.0f, result);
        }

        public override void Norm1(Volume<float> result)
        {
            this.Reduce(result, cudnnReduceTensorOp.Norm1);
        }

        private void Op(Volume<float> right, cudnnOpTensorOp op, Volume<float> result)
        {
            if (!(result.Storage is VolumeStorage resultStorage))
            {
                throw new ArgumentException($"{nameof(result)} storage should be VolumeStorage", nameof(result));
            }

            VolumeStorage rightStorage = null;
            if (right != null)
            {
                rightStorage = right.Storage as VolumeStorage;
                if (rightStorage == null)
                {
                    throw new ArgumentException($"{nameof(right)} storage should be VolumeStorage", nameof(right));
                }
            }

            // Copy to device if not already done
            this._volumeStorage.CopyToDevice();
            rightStorage?.CopyToDevice();
            resultStorage.CopyToDevice();

            var aStorage = this._volumeStorage;
            Shape bShape = null;
            VolumeStorage bStorage = null;
            if (rightStorage != null)
            {
                bStorage = rightStorage;
                if (bStorage.Shape.TotalLength > aStorage.Shape.TotalLength)
                {
                    aStorage = rightStorage;
                    bStorage = this._volumeStorage;
                }

                bShape = bStorage.Shape;
            }

            var n = aStorage.Shape.Dimensions[3];
            var c = aStorage.Shape.Dimensions[2];
            var h = aStorage.Shape.Dimensions[1];
            var w = aStorage.Shape.Dimensions[0];

            // Add tensors
            using var descA = new TensorDescriptor();
            using var descB = new TensorDescriptor();
            using var descC = new TensorDescriptor();

            descA.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
            if (bShape != null)
            {
                descB.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                    bShape.Dimensions[3],
                    bShape.Dimensions[2],
                    bShape.Dimensions[1],
                    bShape.Dimensions[0]);
            }

            descC.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);

            using var opt = new OpTensorDescriptor(this._context.CudnnContext);

            opt.SetOpTensorDescriptor(op, cudnnDataType.Float, cudnnNanPropagation.PropagateNan);

            var one = 1.0f;
            var zero = 0.0f;

            var status = CudaDNNNativeMethods.cudnnOpTensor(
                this._context.CudnnContext.Handle,
                opt.Desc,
                ref one, descA.Desc, aStorage.DeviceBuffer.DevicePointer,
                ref one, bStorage != null ? descB.Desc : descA.Desc, bStorage?.DeviceBuffer.DevicePointer ?? aStorage.DeviceBuffer.DevicePointer,
                ref zero, descC.Desc, resultStorage.DeviceBuffer.DevicePointer);

            if (status != cudnnStatus.Success)
            {
                throw new Exception(CudaDNNNativeMethods.cudnnGetErrorString(status));
            }

            resultStorage.Location = DataLocation.Device;
        }

        public override void Pool(int windowWidth, int windowHeight,
            int horizontalPad, int verticalPad, int horizontalStride, int verticalStride, Volume<float> result)
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
            using var poolingDesc = new PoolingDescriptor();
            using var srcDesc = new TensorDescriptor();
            using var resultDesc = new TensorDescriptor();

            srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                this.Shape.Dimensions[3],
                this.Shape.Dimensions[2],
                this.Shape.Dimensions[1],
                this.Shape.Dimensions[0]);

            resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                result.Shape.Dimensions[3],
                result.Shape.Dimensions[2],
                result.Shape.Dimensions[1],
                result.Shape.Dimensions[0]);

            poolingDesc.SetPooling2dDescriptor(cudnnPoolingMode.Max, cudnnNanPropagation.NotPropagateNan,
                windowHeight, windowWidth,
                verticalPad, horizontalPad, verticalStride, horizontalStride);

            this._context.CudnnContext.PoolingForward(poolingDesc, 1.0f, srcDesc, this._volumeStorage.DeviceBuffer, 0.0f,
                resultDesc, resultStorage.DeviceBuffer);
        }

        public override void PoolGradient(Volume<float> input, Volume<float> outputGradient,
            int windowWidth, int windowHeight,
            int horizontalPad, int verticalPad, int horizontalStride, int verticalStride,
            Volume<float> inputGradient)
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

            using var poolingDesc = new PoolingDescriptor();
            using var srcDesc = new TensorDescriptor();
            using var srcDiffDesc = new TensorDescriptor();
            using var destDesc = new TensorDescriptor();
            using var destDiffDesc = new TensorDescriptor();

            var n = this.Shape.Dimensions[3];
            var c = this.Shape.Dimensions[2];
            var h = this.Shape.Dimensions[1];
            var w = this.Shape.Dimensions[0];

            srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
            srcDiffDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);

            destDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                inputStorage.Shape.Dimensions[3],
                inputStorage.Shape.Dimensions[2],
                inputStorage.Shape.Dimensions[1],
                inputStorage.Shape.Dimensions[0]);
            destDiffDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                inputStorage.Shape.Dimensions[3],
                inputStorage.Shape.Dimensions[2],
                inputStorage.Shape.Dimensions[1],
                inputStorage.Shape.Dimensions[0]);

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

        public override void Power(Volume<float> power, Volume<float> result)
        {
            if (!Equals(this.Shape, power.Shape) && power.Shape.TotalLength > 1)
            {
                throw new ArgumentException("this volume and power should have the same shape OR power should be a scalar.");
            }

            if (!Equals(this.Shape, result.Shape))
            {
                throw new ArgumentException($"this volume and result volume should have the same shape ({this.Shape} != {result.Shape})");
            }

            _kernelLoader.RunKernel("power", this, power, result, power.Shape.TotalLength == 1 ? 1 : 0);
        }

        public override void Reduce(TensorReduceOp op, Volume<float> result)
        {
            this.Reduce(result, op.ToCudnn());
        }

        private void Reduce(Volume<float> result, cudnnReduceTensorOp op)
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

            using var reduceTensorDesc = new ReduceTensorDescriptor();
            using var aDesc = new TensorDescriptor();
            using var cDesc = new TensorDescriptor();

            var an = this.Shape.Dimensions[3];
            var ac = this.Shape.Dimensions[2];
            var ah = this.Shape.Dimensions[1];
            var aw = this.Shape.Dimensions[0];

            var cn = result.Shape.Dimensions[3];
            var cc = result.Shape.Dimensions[2];
            var ch = result.Shape.Dimensions[1];
            var cw = result.Shape.Dimensions[0];

            aDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, an, ac, ah, aw);
            cDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, cn, cc, ch, cw);

            reduceTensorDesc.SetReduceTensorDescriptor(op, cudnnDataType.Float, cudnnNanPropagation.NotPropagateNan, cudnnReduceTensorIndices.NoIndices,
                cudnnIndicesType.Indices32Bit);

            var workspaceSize = this._context.CudnnContext.GetReductionWorkspaceSize(reduceTensorDesc, aDesc, cDesc);
            workspaceSize = workspaceSize == 0 ? new SizeT(1) : workspaceSize;

            if (cStorage.ReductionStorage == null || cStorage.ReductionStorage.Size != workspaceSize)
            {
                Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "DoReduce", workspaceSize);
                cStorage.ReductionStorage = new CudaDeviceVariable<byte>(workspaceSize);
            }

            this._context.CudnnContext.ReduceTensor(reduceTensorDesc,
                CudaDeviceVariable<uint>.Null,
                cStorage.ReductionStorage,
                cStorage.ReductionStorage.SizeInBytes,
                1.0f, aDesc, aStorage.DeviceBuffer,
                0.0f, cDesc, cStorage.DeviceBuffer);
        }

        public override void Relu(Volume<float> result)
        {
            this.Activation(result, cudnnActivationMode.Relu);
        }

        public override void ReluGradient(Volume<float> input, Volume<float> outputGradient,
            Volume<float> inputGradient)
        {
            this.ActivationGradient(input, outputGradient, inputGradient, cudnnActivationMode.Relu);
        }

        public override void Sigmoid(Volume<float> result)
        {
            this.Activation(result, cudnnActivationMode.Sigmoid);
        }

        public override void SigmoidGradient(Volume<float> input, Volume<float> outputGradient,
            Volume<float> inputGradient)
        {
            this.ActivationGradient(input, outputGradient, inputGradient, cudnnActivationMode.Sigmoid);
        }

        public override void Softmax(Volume<float> output)
        {
            var inputStorage = this._volumeStorage;
            var outputStorage = output.Storage as VolumeStorage;

            // Copy to device if not already done
            inputStorage.CopyToDevice();
            outputStorage.CopyToDevice();

            using var srcDesc = new TensorDescriptor();
            using var destDesc = new TensorDescriptor();

            var n = this.Shape.Dimensions[3];
            var c = this.Shape.Dimensions[2];
            var h = this.Shape.Dimensions[1];
            var w = this.Shape.Dimensions[0];

            srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
            destDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);

            this._context.CudnnContext.SoftmaxForward(cudnnSoftmaxAlgorithm.Accurate, cudnnSoftmaxMode.Channel, 1.0f,
                srcDesc, inputStorage.DeviceBuffer, 0.0f,
                destDesc, outputStorage.DeviceBuffer);
        }

        public override void SoftmaxGradient(Volume<float> outputGradient, Volume<float> inputGradient)
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

            using var srcDesc = new TensorDescriptor();
            using var srcDiffDesc = new TensorDescriptor();
            using var destDiffDesc = new TensorDescriptor();

            var n = this.Shape.Dimensions[3];
            var c = this.Shape.Dimensions[2];
            var h = this.Shape.Dimensions[1];
            var w = this.Shape.Dimensions[0];

            srcDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
            srcDiffDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);
            destDiffDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, n, c, h, w);

            this._context.CudnnContext.SoftmaxBackward(cudnnSoftmaxAlgorithm.Accurate, cudnnSoftmaxMode.Channel, 1.0f,
                srcDesc, outputStorage.DeviceBuffer,
                srcDiffDesc, outputGradientStorage.DeviceBuffer,
                0.0f,
                destDiffDesc, inputGradientStorage.DeviceBuffer);
        }

        public override void Sqrt(Volume<float> result)
        {
            this.Op(null, cudnnOpTensorOp.OpTensorSqrt, result);
        }

        public override void SubtractFrom(Volume<float> other, Volume<float> result)
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
            using var subtractorDesc = new TensorDescriptor();
            using var resultDesc = new TensorDescriptor();

            subtractorDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                other.Shape.Dimensions[3],
                other.Shape.Dimensions[2],
                other.Shape.Dimensions[1],
                other.Shape.Dimensions[0]);

            resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float,
                this.Shape.Dimensions[3],
                this.Shape.Dimensions[2],
                this.Shape.Dimensions[1],
                this.Shape.Dimensions[0]);

            this._context.CudnnContext.AddTensor(
                -1.0f, subtractorDesc, this._volumeStorage.DeviceBuffer,
                1.0f, resultDesc, resultStorage.DeviceBuffer);
        }

        public override void Sum(Volume<float> result)
        {
            this.Reduce(TensorReduceOp.Add, result);
        }

        public override void Tanh(Volume<float> result)
        {
            this.Activation(result, cudnnActivationMode.Tanh);
        }

        public override void TanhGradient(Volume<float> input, Volume<float> outputGradient,
            Volume<float> inputGradient)
        {
            this.ActivationGradient(input, outputGradient, inputGradient, cudnnActivationMode.Tanh);
        }

        public override void Tile(Volume<float> reps, Volume<float> result)
        {
            _kernelLoader.RunKernel("tile", this, result
                , new object[]
                {
                    this.Shape.Dimensions[0], this.Shape.Dimensions[1], this.Shape.Dimensions[2], this.Shape.Dimensions[3], result.Shape.Dimensions[0], result.Shape.Dimensions[1],
                    result.Shape.Dimensions[2], result.Shape.Dimensions[3]
                });
        }

        public override void Transpose(Volume<float> result)
        {
            var inputStorage = this.Storage as VolumeStorage;
            var resultStorage = result.Storage as VolumeStorage;
            inputStorage.CopyToDevice();
            resultStorage.CopyToDevice();

            var m = this.Shape.Dimensions[1];
            var n = this.Shape.Dimensions[0];

            for (var b = 0; b < result.Shape.Dimensions[3]; b++) // for each batch
            {
                this._context.CudaBlasHandle.Geam(
                    Operation.Transpose,
                    Operation.NonTranspose, m, n,
                    1.0f,
                    new CudaDeviceVariable<float>(inputStorage.DeviceBuffer.DevicePointer + b * m * n * sizeof(float)), n,
                    new CudaDeviceVariable<float>(inputStorage.DeviceBuffer.DevicePointer + b * m * n * sizeof(float)), m, // not used because Beta = 0
                    0.0f,
                    new CudaDeviceVariable<float>(resultStorage.DeviceBuffer.DevicePointer + b * m * n * sizeof(float)), m);
            }
        }
    }
}