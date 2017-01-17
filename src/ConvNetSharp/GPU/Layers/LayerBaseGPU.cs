using ConvNetSharp.Layers;
using ManagedCuda;
using ManagedCuda.NVRTC;
using System;
using System.Diagnostics;
using System.IO;

namespace ConvNetSharp.GPU.Layers
{
    public unsafe abstract class LayerBaseGPU : LayerBase
    {
        static CudaContext ctx;

        static LayerBaseGPU()
        {
            ctx = new CudaContext(0, true);
        }

        protected int defaultBlockCount, defaultThreadsPerBlock, warpSize;

        protected CudaStream defaultStream;
        protected CudaKernel kernel;
        string path;

        public void Dispose() => Dispose(true);

        ~LayerBaseGPU() { Dispose(false); } // we don't want to see this... this means we failed

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                GC.SuppressFinalize(this); // dispose was called correctly
            }

            if (disposing) // clean up managed resources
            {
                Dispose(ref defaultStream);
                Dispose(ref ctx);
            }
        }

        // utility method to dispose and wipe fields
        protected void Dispose<T>(ref T field) where T : class, IDisposable
        {
            if (field != null)
            {
                try { field.Dispose(); } catch (Exception ex) { Debug.WriteLine(ex.Message); }
                field = null;
            }
        }

        public LayerBaseGPU(string path)
        {
            this.path = path;

            // note that this initializes a lot of things and binds *to the thread*

            var props = ctx.GetDeviceInfo();
            defaultBlockCount = props.MultiProcessorCount * 32;
            defaultThreadsPerBlock = props.MaxThreadsPerBlock;
            warpSize = props.WarpSize;
        }

        public virtual void InitializeData(params int[] parameters)
        {
            // allocate a stream for async/overlapped operations; note we're only going to use
            // one stream, but in complex code you can use different streams to allow concurrent
            // memory transfer while unrelated kernels execute, etc
            this.defaultStream = new CudaStream();
        }

        protected static int RoundUp(int value, int blockSize)
        {
            if ((value % blockSize) != 0)
            {   // take away the surplus, and add an entire extra block
                value += blockSize - (value % blockSize);
            }
            return value;
        }

        internal void Synchronize()
        {
            ctx.Synchronize(); // this synchronizes (waits for) **all streams**
            // to synchronize a single stream, use {theStream}.Synchronize();
        }

        internal nvrtcResult LoadKernel(out string log)
        {
            nvrtcResult result;
            using (var rtc = new CudaRuntimeCompiler(File.ReadAllText(this.path), Path.GetFileName(this.path)))
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
                    byte[] ptx = rtc.GetPTX();
                    kernel = ctx.LoadKernelFatBin(ptx, "Run"); // hard-coded method name from the CUDA kernel
                }
            }
            return result;
        }
    }
}
