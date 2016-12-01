using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using ConvNetSharp.Layers;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace ConvNetSharp.GPU.Layers
{
    public unsafe class ConvLayerGPU : LayerBaseGPU, IDotProductLayer
    {
        IntPtr hostBiasesPointer;
        double* hostBiasesBuffer;
        CudaDeviceVariable<double> deviceBiasesBuffer;

        IntPtr hostFilterGradientPointer;
        double* hostFilterGradientBuffer;
        CudaDeviceVariable<double> deviceFilterGradientBuffer;

        IntPtr hostFilterPointer;
        double* hostFilterBuffer;
        CudaDeviceVariable<double> deviceFilterBuffer;

        IntPtr hostInputPointer;
        double* hostInputBuffer;
        CudaDeviceVariable<double> deviceInputBuffer;

        IntPtr hostInputGradientPointer;
        double* hostInputGradientBuffer;
        CudaDeviceVariable<double> deviceInputGradientBuffer;

        IntPtr hostOutputGradientPointer;
        double* hostOutputGradientBuffer;
        CudaDeviceVariable<double> deviceOutputGradientBuffer;

        IntPtr hostOutputPointer;
        double* hostOutputBuffer;
        CudaDeviceVariable<double> deviceOutputBuffer;

        CudaKernel forwardKernel;
        CudaKernel backwardFilterKernel;
        CudaKernel backwardInputKernel;

        public ConvLayerGPU(ConvLayer convLayer) : this(convLayer.Width, convLayer.Height, convLayer.FilterCount)
        {
            this.Biases = convLayer.Biases;
            this.Filters = convLayer.Filters;
            this.Stride = convLayer.Stride;
            this.Pad = convLayer.Pad;
        }

        public ConvLayerGPU(int width, int height, int filterCount) : base()
        {
            this.L1DecayMul = 0.0;
            this.L2DecayMul = 1.0;

            this.FilterCount = filterCount;
            this.Width = width;
            this.Height = height;

            string log;
            this.LoadKernel(@".\GPU\Kernels\convolution_forward.cu", out this.forwardKernel, out log);
            if (!string.IsNullOrEmpty(log))
            {
                throw new Exception();
            }

            this.LoadKernel(@".\GPU\Kernels\convolution_backward_filter.cu", out this.backwardFilterKernel, out log);
            if (!string.IsNullOrEmpty(log))
            {
                throw new Exception();
            }

            this.LoadKernel(@".\GPU\Kernels\convolution_backward_input.cu", out this.backwardInputKernel, out log);
            if (!string.IsNullOrEmpty(log))
            {
                throw new Exception();
            }
        }

        public int Width { get; private set; }

        public int Height { get; private set; }

        public Volume Biases { get; private set; }

        public List<Volume> Filters { get; private set; }

        public int FilterCount { get; private set; }

        public double L1DecayMul { get; set; }

        public double L2DecayMul { get; set; }

        public int Stride { get; set; } = 1;

        public int Pad { get; set; }

        public double BiasPref { get; set; }

        public override IVolume Forward(IVolume input, bool isTraining = false)
        {
            this.InputActivation = input;

            this.FillForward();

            this.RunForwardAsync();

            this.OutputActivation = new Volume(this.OutputWidth, this.OutputHeight, this.OutputDepth, 0.0);

            this.CopyToHost();

            this.Synchronize();

            this.FillOutput();

            return this.OutputActivation;
        }

        public override void Backward()
        {
            var volume = this.InputActivation;
            volume.ZeroGradients(); // zero out gradient wrt bottom data, we're about to fill it

            // Bias gradient
            var task = Task.Factory.StartNew(() =>
            {
                Parallel.For(0, this.OutputDepth, (depth) =>
                {
                    var y = -this.Pad;
                    for (var ay = 0; ay < this.OutputHeight; y += this.Stride, ay++)
                    {
                        // xyStride
                        var x = -this.Pad;
                        for (var ax = 0; ax < this.OutputWidth; x += this.Stride, ax++)
                        {
                            // xyStride

                            // convolve centered at this particular location
                            var chainGradient = this.OutputActivation.GetGradient(ax, ay, depth);
                            this.Biases.SetGradient(depth, this.Biases.GetGradient(depth) + chainGradient);
                        }
                    }
                });
            });

            this.FillBackward();

            // Filter gradient
            this.RunFilterBackwardAsync();
            this.CopyFilterGradientToHost();

            this.Synchronize();

            this.FillFilterGradient();

            // Input gradient
            this.RunInputBackwardAsync();
            this.CopyInputGradientToHost();

            this.Synchronize();

            this.FillInputGradient();

            task.Wait();
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            this.UpdateOutputSize();
        }

        public void UpdateOutputSize()
        {
            // required
            this.OutputDepth = this.FilterCount;

            // computed
            // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
            // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
            // final application.
            this.OutputWidth = (int)Math.Floor((this.InputWidth + this.Pad * 2 - this.Width) / (double)this.Stride + 1);
            this.OutputHeight = (int)Math.Floor((this.InputHeight + this.Pad * 2 - this.Height) / (double)this.Stride + 1);

            // initializations
            var bias = this.BiasPref;
            this.Filters = new List<Volume>();

            for (var i = 0; i < this.OutputDepth; i++)
            {
                this.Filters.Add(new Volume(this.Width, this.Height, this.InputDepth));
            }

            this.Biases = new Volume(1, 1, this.OutputDepth, bias);

            this.InitializeData();
        }

        public override List<ParametersAndGradients> GetParametersAndGradients()
        {
            var response = new List<ParametersAndGradients>();
            for (var i = 0; i < this.OutputDepth; i++)
            {
                response.Add(new ParametersAndGradients
                {
                    //Parameters = this.Filters[i].Weights,
                    //Gradients = this.Filters[i].WeightGradients,
                    Volume = this.Filters[i],
                    L2DecayMul = this.L2DecayMul,
                    L1DecayMul = this.L1DecayMul
                });
            }

            response.Add(new ParametersAndGradients
            {
                //Parameters = this.Biases.Weights,
                //Gradients = this.Biases.WeightGradients,
                Volume = this.Biases,
                L1DecayMul = 0.0,
                L2DecayMul = 0.0
            });

            return response;
        }

        public override void InitializeData(params int[] parameters)
        {
            // Host Biases
            this.hostBiasesPointer = IntPtr.Zero;
            var res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref this.hostBiasesPointer, this.OutputDepth * sizeof(double));
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
            this.hostBiasesBuffer = (double*)this.hostBiasesPointer;
            // Device Biases
            this.deviceBiasesBuffer = new CudaDeviceVariable<double>(this.OutputDepth);

            // Host Filters gradient
            this.hostFilterGradientPointer = IntPtr.Zero;
            res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref this.hostFilterGradientPointer, this.FilterCount * this.Width * this.Height * this.InputDepth * sizeof(double));
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
            this.hostFilterGradientBuffer = (double*)this.hostFilterGradientPointer;
            // Device Filters gradient
            this.deviceFilterGradientBuffer = new CudaDeviceVariable<double>(this.FilterCount * this.Width * this.Height * this.InputDepth);

            // Host Filters 
            this.hostFilterPointer = IntPtr.Zero;
            res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref this.hostFilterPointer, this.FilterCount * this.Width * this.Height * this.InputDepth * sizeof(double));
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
            this.hostFilterBuffer = (double*)this.hostFilterPointer;
            // Device Filters 
            this.deviceFilterBuffer = new CudaDeviceVariable<double>(this.FilterCount * this.Width * this.Height * this.InputDepth);

            // Host input
            this.hostInputPointer = IntPtr.Zero;
            res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref this.hostInputPointer, this.InputWidth * this.InputHeight * this.InputDepth * sizeof(double));
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
            this.hostInputBuffer = (double*)this.hostInputPointer;
            // Device input
            this.deviceInputBuffer = new CudaDeviceVariable<double>(this.InputWidth * this.InputHeight * this.InputDepth);

            // Host input gradients
            this.hostInputGradientPointer = IntPtr.Zero;
            res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref this.hostInputGradientPointer, this.InputWidth * this.InputHeight * this.InputDepth * sizeof(double));
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
            this.hostInputGradientBuffer = (double*)this.hostInputGradientPointer;
            // Device input gradients
            this.deviceInputGradientBuffer = new CudaDeviceVariable<double>(this.InputWidth * this.InputHeight * this.InputDepth);

            // Host output gradients
            this.hostOutputGradientPointer = IntPtr.Zero;
            res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref this.hostOutputGradientPointer, this.OutputWidth * this.OutputHeight * this.OutputDepth * sizeof(double));
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
            this.hostOutputGradientBuffer = (double*)this.hostOutputGradientPointer;
            // Device output gradients
            this.deviceOutputGradientBuffer = new CudaDeviceVariable<double>(this.OutputWidth * this.OutputHeight * this.OutputDepth);

            // Host output
            this.hostOutputPointer = IntPtr.Zero;
            res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref this.hostOutputPointer, this.OutputWidth * this.OutputHeight * this.OutputDepth * sizeof(double));
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
            this.hostOutputBuffer = (double*)this.hostOutputPointer;
            // Device output
            this.deviceOutputBuffer = new CudaDeviceVariable<double>(this.OutputWidth * this.OutputHeight * this.OutputDepth);

            base.InitializeData(parameters);
        }

        public void FillForward()
        {
            // Fill up biases
            for (int i = 0; i < this.Biases.Length; i++)
            {
                this.hostBiasesBuffer[i] = this.Biases.Get(i);
            }

            var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(deviceBiasesBuffer.DevicePointer, hostBiasesPointer, deviceBiasesBuffer.SizeInBytes, defaultStream.Stream);
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }

            // Fill up filters
            var count = this.Width * this.Height * this.InputDepth;
            for (int j = 0; j < this.FilterCount; j++)
            {
                for (int i = 0; i < this.Filters[j].Length; i++)
                {
                    this.hostFilterBuffer[i + j * count] = this.Filters[j].Get(i);
                }
            }

            res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(deviceFilterBuffer.DevicePointer, hostFilterPointer, deviceFilterBuffer.SizeInBytes, defaultStream.Stream);
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }

            // Fill up input
            for (int i = 0; i < this.InputActivation.Length; i++)
            {
                this.hostInputBuffer[i] = this.InputActivation.Get(i);
            }

            res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(deviceInputBuffer.DevicePointer, hostInputPointer, deviceInputBuffer.SizeInBytes, defaultStream.Stream);
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
        }

        public void FillBackward()
        {
            // Fill up input
            for (int i = 0; i < this.InputActivation.Length; i++)
            {
                this.hostInputBuffer[i] = this.InputActivation.Get(i);
            }

            var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(deviceInputBuffer.DevicePointer, hostInputPointer, deviceInputBuffer.SizeInBytes, defaultStream.Stream);
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }

            // Fill up input gradients
            for (int i = 0; i < this.InputActivation.Length; i++)
            {
                this.hostInputGradientBuffer[i] = this.InputActivation.GetGradient(i);
            }

            res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(deviceInputGradientBuffer.DevicePointer, hostInputGradientPointer, deviceInputGradientBuffer.SizeInBytes, defaultStream.Stream);
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }

            // Fill up output gradients
            for (int i = 0; i < this.OutputActivation.Length; i++)
            {
                this.hostOutputGradientBuffer[i] = this.OutputActivation.GetGradient(i);
            }

            res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(deviceOutputGradientBuffer.DevicePointer, hostOutputGradientPointer, deviceOutputGradientBuffer.SizeInBytes, defaultStream.Stream);
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }

            // Fill up filters
            var count = this.Width * this.Height * this.InputDepth;
            for (int j = 0; j < this.FilterCount; j++)
            {
                for (int i = 0; i < count; i++)
                {
                    this.hostFilterGradientBuffer[i + j * count] = this.Filters[j].GetGradient(i);
                }
            }

            res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(deviceFilterGradientBuffer.DevicePointer, hostFilterGradientPointer, deviceFilterGradientBuffer.SizeInBytes, defaultStream.Stream);
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
        }

        private void RunFilterBackwardAsync(params object[] parameters)
        {
            var count = this.Height * this.Width * this.InputDepth * this.OutputDepth;

            // configure the dimensions; note, usually this is a lot more dynamic based
            // on input data, but we'll still go through the motions
            int threadsPerBlock, blockCount;
            if (count <= defaultThreadsPerBlock) // a single block
            {
                blockCount = 1;
                threadsPerBlock = RoundUp(count, warpSize); // slight caveat here; if you are using "shuffle" operations, you
                                                            // need to use entire "warp"s - otherwise the result is undefined
            }
            else if (count >= defaultThreadsPerBlock * defaultBlockCount)
            {
                // more than enough work to keep us busy; just use that
                threadsPerBlock = defaultThreadsPerBlock;
                blockCount = defaultBlockCount;
            }
            else
            {
                // do the math to figure out how many blocks we need
                threadsPerBlock = defaultThreadsPerBlock;
                blockCount = (count + threadsPerBlock - 1) / threadsPerBlock;
            }

            // we're using 1-D math, but actually CUDA supports blocks and grids that span 3 dimensions
            this.backwardFilterKernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(threadsPerBlock, 1, 1);
            this.backwardFilterKernel.GridDimensions = new ManagedCuda.VectorTypes.dim3(blockCount, 1, 1);

            // invoke the kernel
            this.backwardFilterKernel.RunAsync(this.defaultStream.Stream, new object[] {
                this.Pad, this.Stride,
                this.Width, this.Height, this.InputDepth, this.deviceFilterGradientBuffer.DevicePointer,
                this.InputWidth, this.InputHeight, this.InputDepth, this.deviceInputBuffer.DevicePointer,
                this.OutputWidth, this.OutputHeight, this.OutputDepth, this.deviceOutputGradientBuffer.DevicePointer});
        }

        private void RunInputBackwardAsync(params object[] parameters)
        {
            var count = this.InputWidth * this.InputHeight * this.InputDepth;

            // configure the dimensions; note, usually this is a lot more dynamic based
            // on input data, but we'll still go through the motions
            int threadsPerBlock, blockCount;
            if (count <= defaultThreadsPerBlock) // a single block
            {
                blockCount = 1;
                threadsPerBlock = RoundUp(count, warpSize); // slight caveat here; if you are using "shuffle" operations, you
                                                            // need to use entire "warp"s - otherwise the result is undefined
            }
            else if (count >= defaultThreadsPerBlock * defaultBlockCount)
            {
                // more than enough work to keep us busy; just use that
                threadsPerBlock = defaultThreadsPerBlock;
                blockCount = defaultBlockCount;
            }
            else
            {
                // do the math to figure out how many blocks we need
                threadsPerBlock = defaultThreadsPerBlock;
                blockCount = (count + threadsPerBlock - 1) / threadsPerBlock;
            }

            // we're using 1-D math, but actually CUDA supports blocks and grids that span 3 dimensions
            this.backwardInputKernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(threadsPerBlock, 1, 1);
            this.backwardInputKernel.GridDimensions = new ManagedCuda.VectorTypes.dim3(blockCount, 1, 1);

            // invoke the kernel
            this.backwardInputKernel.RunAsync(this.defaultStream.Stream, new object[] {
                this.Pad, this.Stride,
                this.Width, this.Height, this.InputDepth, this.deviceFilterBuffer.DevicePointer,
                this.InputWidth, this.InputHeight, this.InputDepth, this.deviceInputGradientBuffer.DevicePointer,
                this.OutputWidth, this.OutputHeight, this.OutputDepth, this.deviceOutputGradientBuffer.DevicePointer});
        }

        private void RunForwardAsync(params object[] parameters)
        {
            var count = this.OutputHeight * this.OutputWidth * this.OutputDepth;

            // configure the dimensions; note, usually this is a lot more dynamic based
            // on input data, but we'll still go through the motions
            int threadsPerBlock, blockCount;
            if (count <= defaultThreadsPerBlock) // a single block
            {
                blockCount = 1;
                threadsPerBlock = RoundUp(count, warpSize); // slight caveat here; if you are using "shuffle" operations, you
                                                            // need to use entire "warp"s - otherwise the result is undefined
            }
            else if (count >= defaultThreadsPerBlock * defaultBlockCount)
            {
                // more than enough work to keep us busy; just use that
                threadsPerBlock = defaultThreadsPerBlock;
                blockCount = defaultBlockCount;
            }
            else
            {
                // do the math to figure out how many blocks we need
                threadsPerBlock = defaultThreadsPerBlock;
                blockCount = (count + threadsPerBlock - 1) / threadsPerBlock;
            }

            // we're using 1-D math, but actually CUDA supports blocks and grids that span 3 dimensions
            this.forwardKernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(threadsPerBlock, 1, 1);
            this.forwardKernel.GridDimensions = new ManagedCuda.VectorTypes.dim3(blockCount, 1, 1);

            // invoke the kernel
            this.forwardKernel.RunAsync(this.defaultStream.Stream, new object[] {
                this.Pad, this.Stride, this.deviceBiasesBuffer.DevicePointer,
                this.Width, this.Height, this.InputDepth, this.deviceFilterBuffer.DevicePointer,
                this.InputWidth, this.InputHeight, this.InputDepth, this.deviceInputBuffer.DevicePointer,
                this.OutputWidth, this.OutputHeight, this.OutputDepth, this.deviceOutputBuffer.DevicePointer});
        }

        private void CopyToHost()
        {
            // Output
            var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(new IntPtr(this.hostOutputBuffer), this.deviceOutputBuffer.DevicePointer, this.deviceOutputBuffer.SizeInBytes, defaultStream.Stream);
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
        }

        private void CopyFilterGradientToHost()
        {
            var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(new IntPtr(this.hostFilterGradientBuffer), this.deviceFilterGradientBuffer.DevicePointer, this.deviceFilterGradientBuffer.SizeInBytes, defaultStream.Stream);
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
        }

        private void CopyInputGradientToHost()
        {
            var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(new IntPtr(this.hostInputGradientBuffer), this.deviceInputGradientBuffer.DevicePointer, this.deviceInputGradientBuffer.SizeInBytes, defaultStream.Stream);
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
        }

        private void FillOutput()
        {
            // Fill up output
            for (int i = 0; i < this.OutputActivation.Length; i++)
            {
                this.OutputActivation.Set(i, this.hostOutputBuffer[i]);
            }
        }

        private void FillFilterGradient()
        {
            var length = this.Height * this.Width * this.InputDepth;
            // Fill up output
            for (int i = 0; i < this.FilterCount; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    this.Filters[i].SetGradient(j, this.hostFilterGradientBuffer[j + i * length]);
                }
            }
        }

        private void FillInputGradient()
        {
            var length = this.InputWidth * this.InputHeight * this.InputDepth;
            // Fill up output
            for (int i = 0; i < length; i++)
            {
                this.InputActivation.SetGradient(i, this.hostInputGradientBuffer[i]);
            }
        }
    }
}

