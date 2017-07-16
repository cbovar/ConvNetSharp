using System;
using System.Collections.Generic;
using System.IO;
using ManagedCuda;
using ManagedCuda.NVRTC;
using ManagedCuda.VectorTypes;

namespace ConvNetSharp.Volume.GPU
{
    internal class KernelLoader<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly GpuContext _context;

        private readonly Dictionary<string, CudaKernel> _kernels = new Dictionary<string, CudaKernel>();

        public KernelLoader(GpuContext context)
        {
            this._context = context;
        }

        private void AddKernel(string name, CudaKernel kernel)
        {
            this._kernels[name] = kernel;
        }

        public void LoadKernel(string name, Stream stream)
        {
            using (StreamReader reader = new StreamReader(stream))
            {
                string result = reader.ReadToEnd();
                File.WriteAllText($"{name}.cu", result);

                this.LoadKernel(name, $"{name}.cu");
            }
        }

        public void LoadKernel(string name, string path)
        {
            CudaKernel kernel;
            string log;
            var result = LoadKernel(path, out kernel, out log);
            if (result == nvrtcResult.Success)
            {
                AddKernel(name, kernel);
            }
        }

        private nvrtcResult LoadKernel(string path, out CudaKernel kernel, out string log)
        {
            nvrtcResult result;
            using (var rtc = new CudaRuntimeCompiler(File.ReadAllText(path), Path.GetFileName(path)))
            {
                try
                {
                    rtc.Compile(new string[0]); // see http://docs.nvidia.com/cuda/nvrtc/index.html for usage and options
                    result = nvrtcResult.Success;
                }
                catch (NVRTCException ex)
                {
                    result = ex.NVRTCError;
                }

                log = rtc.GetLogAsString();

                if (result == nvrtcResult.Success)
                {
                    var ptx = rtc.GetPTX();
                    kernel = this._context.CudaContext.LoadKernelFatBin(ptx, "Run"); // hard-coded method name from the CUDA kernel
                }
                else
                {
                    kernel = null;
                }
            }
            return result;
        }

        private static int RoundUp(int value, int blockSize)
        {
            if (value % blockSize != 0)
            {
                // take away the surplus, and add an entire extra block
                value += blockSize - value % blockSize;
            }
            return value;
        }

        public void RunKernel(string kernelName, Volume<T> input, Volume<T> output)
        {
            CudaKernel kernel;
            if (this._kernels.TryGetValue(kernelName, out kernel))
            {
                RunKernel(input, output, kernel);
            }
            else
            {
                throw new ArgumentException($"Could not find kernel '{kernelName}'", nameof(kernelName));
            }
        }

        private void RunKernel(Volume<T> input, Volume<T> output, CudaKernel kernel)
        {
            if (!Equals(input.Shape, output.Shape))
            {
                throw new ArgumentException($"{nameof(input)} and {nameof(output)} should have the same shape.");
            }

            var inputStorage = input.Storage as IVolumeStorage<T>;
            if (inputStorage == null)
            {
                throw new ArgumentException($"{nameof(input)} storage should be VolumeStorage", nameof(input));
            }

            var outputStorage = output.Storage as IVolumeStorage<T>;
            if (outputStorage == null)
            {
                throw new ArgumentException($"{nameof(output)} storage should be VolumeStorage", nameof(output));
            }

            inputStorage.CopyToDevice();
            outputStorage.CopyToDevice();

            var count = (int) input.Shape.TotalLength;

            // configure the dimensions; note, usually this is a lot more dynamic based
            // on input data, but we'll still go through the motions
            int threadsPerBlock, blockCount;
            if (count <= this._context.DefaultThreadsPerBlock) // a single block
            {
                blockCount = 1;
                threadsPerBlock = RoundUp(count, this._context.WarpSize); // slight caveat here; if you are using "shuffle" operations, you
                // need to use entire "warp"s - otherwise the result is undefined
            }
            else if (count >= this._context.DefaultThreadsPerBlock * this._context.DefaultBlockCount)
            {
                // more than enough work to keep us busy; just use that
                threadsPerBlock = this._context.DefaultThreadsPerBlock;
                blockCount = this._context.DefaultBlockCount;
            }
            else
            {
                // do the math to figure out how many blocks we need
                threadsPerBlock = this._context.DefaultThreadsPerBlock;
                blockCount = (count + threadsPerBlock - 1) / threadsPerBlock;
            }

            // we're using 1-D math, but actually CUDA supports blocks and grids that span 3 dimensions
            kernel.BlockDimensions = new dim3(threadsPerBlock, 1, 1);
            kernel.GridDimensions = new dim3(blockCount, 1, 1);

            // invoke the kernel
            kernel.RunAsync(this._context.DefaultStream.Stream, count, inputStorage.DeviceBuffer.DevicePointer, outputStorage.DeviceBuffer.DevicePointer);
        }
    }
}