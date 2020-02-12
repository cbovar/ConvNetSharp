using System;
using System.Diagnostics;
using ManagedCuda;
using ManagedCuda.CudaBlas;
using ManagedCuda.CudaDNN;

namespace ConvNetSharp.Volume.GPU
{
    public class GpuContext
    {
        private static readonly Lazy<GpuContext> DefaultContextLazy = new Lazy<GpuContext>(() => new GpuContext());
        private CudaContext _cudaContext;
        private CudaStream _defaultStream;

        public GpuContext(int deviceId = 0)
        {
            this.CudaContext = new CudaContext(deviceId, true);
            this.CudaBlasHandle = new CudaBlas();

            var props = this.CudaContext.GetDeviceInfo();
            this.DefaultBlockCount = props.MultiProcessorCount * 32;
            this.DefaultThreadsPerBlock = props.MaxThreadsPerBlock;
            this.WarpSize = props.WarpSize;

            this.DefaultStream = new CudaStream();

            this.CudnnContext = new CudaDNNContext();
        }

        public CudaBlas CudaBlasHandle { get; }

        public CudaContext CudaContext
        {
            get => this._cudaContext;
            private set => this._cudaContext = value;
        }

        public CudaDNNContext CudnnContext { get; }

        public static GpuContext Default => DefaultContextLazy.Value;

        public int DefaultBlockCount { get; }

        public int DefaultThreadsPerBlock { get; }

        public int WarpSize { get; }

        public CudaStream DefaultStream
        {
            get => this._defaultStream;
            set => this._defaultStream = value;
        }

        public void Dispose()
        {
            this.Dispose(true);
        }

        public virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                GC.SuppressFinalize(this);
            }

            if (disposing)
            {
                this.Dispose(ref this._defaultStream);
                this.Dispose(ref this._cudaContext);
            }
        }

        public void Dispose<T>(ref T field) where T : class, IDisposable
        {
            if (field != null)
            {
                try
                {
                    field.Dispose();
                }
                catch (Exception ex)
                {
                    Debug.WriteLine(ex.Message);
                }

                field = null;
            }
        }

        ~GpuContext()
        {
            this.Dispose(false);
        }
    }
}