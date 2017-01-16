using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Threading.Tasks;
using ConvNetSharp.Layers;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace ConvNetSharp.GPU.Layers
{
    [DataContract]
    [Serializable]
    public unsafe class ConvLayerGPU : LayerBaseGPU, IDotProductLayer
    {
        public ConvLayerGPU(int width, int height, int filterCount) : base(@".\GPU\Kernels\convolution.cu")
        {
            this.L1DecayMul = 0.0;
            this.L2DecayMul = 1.0;

            this.FilterCount = filterCount;
            this.Width = width;
            this.Height = height;

            string log;
            this.LoadKernel(out log);

            if (!string.IsNullOrEmpty(log))
            {
                throw new Exception();
            }
        }

        [DataMember]
        public int Width { get; private set; }

        [DataMember]
        public int Height { get; private set; }

        [DataMember]
        public Volume Biases { get; private set; }

        [DataMember]
        public List<Volume> Filters { get; private set; }

        [DataMember]
        public int FilterCount { get; private set; }

        [DataMember]
        public double L1DecayMul { get; set; }

        [DataMember]
        public double L2DecayMul { get; set; }

        [DataMember]
        public int Stride { get; set; } = 1;

        [DataMember]
        public int Pad { get; set; }

        [DataMember]
        public double BiasPref { get; set; }

        public override IVolume Forward(IVolume input, bool isTraining = false)
        {
            this.InputActivation = input;

            this.FillData();

            this.RunAsync();

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

            var volumeWidth = volume.Width;
            var volumeHeight = volume.Height;
            var volumeDepth = volume.Depth;
            var xyStride = this.Stride;

#if PARALLEL
            var locker = new object();
            Parallel.For(0, this.OutputDepth, () => new Volume(volumeWidth, volumeHeight, volumeDepth, 0), (depth, state, temp) =>
#else
            var temp = volume;
            for (var depth = 0; depth < this.OutputDepth; depth++)
#endif
            {
                var filter = this.Filters[depth];
                var y = -this.Pad;
                for (var ay = 0; ay < this.OutputHeight; y += xyStride, ay++)
                {
                    // xyStride
                    var x = -this.Pad;
                    for (var ax = 0; ax < this.OutputWidth; x += xyStride, ax++)
                    {
                        // xyStride

                        // convolve centered at this particular location
                        var chainGradient = this.OutputActivation.GetGradient(ax, ay, depth);
                        // gradient from above, from chain rule
                        for (var fy = 0; fy < filter.Height; fy++)
                        {
                            var oy = y + fy; // coordinates in the original input array coordinates
                            for (var fx = 0; fx < filter.Width; fx++)
                            {
                                var ox = x + fx;
                                if (oy >= 0 && oy < volumeHeight && ox >= 0 && ox < volumeWidth)
                                {
                                    for (var fd = 0; fd < filter.Depth; fd++)
                                    {
                                        filter.AddGradient(fx, fy, fd, volume.Get(ox, oy, fd) * chainGradient);
                                        temp.AddGradient(ox, oy, fd, filter.Get(fx, fy, fd) * chainGradient);
                                    }
                                }
                            }
                        }

                        this.Biases.SetGradient(depth, this.Biases.GetGradient(depth) + chainGradient);
                    }
                }

#if !PARALLEL
            }
#else
                return temp;
            }
                ,
                result =>
                {
                    lock (locker)
                    {
                        volume.AddGradientFrom(result);
                    }
                });
#endif
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            this.UpdateOutputSize();
        }

        internal void UpdateOutputSize()
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

        IntPtr hostBiasesPointer;
        double* hostBiasesBuffer;
        CudaDeviceVariable<double> deviceBiasesBuffer;

        IntPtr hostFiltersPointer;
        double* hostFiltersBuffer;
        CudaDeviceVariable<double> deviceFiltersBuffer;

        IntPtr hostInputPointer;
        double* hostInputBuffer;
        CudaDeviceVariable<double> deviceInputBuffer;

        IntPtr hostOutputPointer;
        double* hostOutputBuffer;
        CudaDeviceVariable<double> deviceOutputBuffer;

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
            this.deviceBiasesBuffer = new CudaDeviceVariable<double>(this.OutputDepth);

            // Host Filters
            this.hostFiltersPointer = IntPtr.Zero;
            res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref this.hostFiltersPointer, this.FilterCount * this.Width * this.Height * this.InputDepth * sizeof(double));
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
            this.hostFiltersBuffer = (double*)this.hostFiltersPointer;
            this.deviceFiltersBuffer = new CudaDeviceVariable<double>(this.FilterCount * this.Width * this.Height * this.InputDepth);

            // Host input
            this.hostInputPointer = IntPtr.Zero;
            res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref this.hostInputPointer, this.InputWidth * this.InputHeight * this.InputDepth * sizeof(double));
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
            this.hostInputBuffer = (double*)this.hostInputPointer;
            this.deviceInputBuffer = new CudaDeviceVariable<double>(this.InputWidth * this.InputHeight * this.InputDepth);

            // Host output
            this.hostOutputPointer = IntPtr.Zero;

            res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref this.hostOutputPointer, this.OutputWidth * this.OutputHeight * this.OutputDepth * sizeof(double));
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
            this.hostOutputBuffer = (double*)this.hostOutputPointer;
            this.deviceOutputBuffer = new CudaDeviceVariable<double>(this.OutputWidth * this.OutputHeight * this.OutputDepth);


            base.InitializeData(parameters);
        }

        public override void FillData()
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
                    this.hostFiltersBuffer[i + j * count] = this.Filters[j].Get(i);
                }
            }

            res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(deviceFiltersBuffer.DevicePointer, hostFiltersPointer, deviceFiltersBuffer.SizeInBytes, defaultStream.Stream);
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

        internal override void RunAsync(params object[] parameters)
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
            this.kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(threadsPerBlock, 1, 1);
            this.kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3(blockCount, 1, 1);

            // invoke the kernel
            this.kernel.RunAsync(this.defaultStream.Stream, new object[] {
                this.Pad, this.Stride, this.deviceBiasesBuffer.DevicePointer,
                this.Width, this.Height, this.InputDepth, this.deviceFiltersBuffer.DevicePointer,
                this.InputWidth, this.InputHeight, this.InputDepth, this.deviceInputBuffer.DevicePointer,
                this.OutputWidth, this.OutputHeight, this.OutputDepth, this.deviceOutputBuffer.DevicePointer});
        }

        internal override void CopyToHost()
        {
            // Output
            var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(new IntPtr(this.hostOutputBuffer), this.deviceOutputBuffer.DevicePointer, this.deviceOutputBuffer.SizeInBytes, defaultStream.Stream);
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
    }
}
