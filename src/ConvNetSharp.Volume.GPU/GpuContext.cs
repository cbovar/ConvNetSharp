using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.CudaBlas;
using ManagedCuda.CudaDNN;

namespace ConvNetSharp.Volume.GPU;

public class GpuContext
{
    private static readonly Lazy<GpuContext> DefaultContextLazy = new(() => new GpuContext());
    private CudaContext _cudaContext;
    private CudaStream _defaultStream;

    public GpuContext(int deviceId = 0)
    {
        // Validate CUDA environment
        this.ValidateCudaEnvironment();

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
        private init => this._cudaContext = value;
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

    private void ValidateCudaEnvironment()
    {
        // Get the installed CUDA driver version
        var driverVersion = CudaContext.GetDriverVersion();

        // Minimum required CUDA version
        Version requiredCudaVersion = new(12, 0);

        // Check the driver version
        if (driverVersion < requiredCudaVersion)
        {
            throw new Exception(
                $"CUDA driver version {driverVersion} is installed. " +
                $"This application requires CUDA {requiredCudaVersion} or newer. Please update your NVIDIA driver.");
        }

        // Get the required cuDNN version
        var requiredCudnnVersion = CudaDNNNativeMethods.Version;

        // Dynamically check the installed cuDNN version
        Version installedCudnnVersion;
        try
        {
            installedCudnnVersion = CudaDNNNativeMethods.cudnnGetVersion();
        }
        catch
        {
            throw new DllNotFoundException(
                "cuDNN library 'cudnn64_7.dll' was not found or could not be loaded. " +
                "Please ensure that the correct version of cuDNN is installed, and the library is accessible in your system PATH.");
        }

        if (installedCudnnVersion < requiredCudnnVersion)
        {
            throw new Exception(
                $"cuDNN version {installedCudnnVersion} is installed. " +
                $"This application requires cuDNN {requiredCudnnVersion} or newer. Please update your cuDNN installation.");
        }

        // Check for other required libraries
        string[] requiredLibraries =
        [
            "cudart64_12.dll", // CUDA Runtime
            "cublas64_12.dll" // cuBLAS
        ];

        foreach (var library in requiredLibraries)
        {
            if (!IsLibraryAvailable(library))
            {
                throw new DllNotFoundException(
                    $"The required CUDA library '{library}' was not found. " +
                    "Please ensure that the NVIDIA CUDA Toolkit 12.0+ is properly installed, " +
                    "and that the CUDA 'bin' directory is added to your system PATH.");
            }
        }
    }

    private static bool IsLibraryAvailable(string libraryName)
    {
        try
        {
            var handle = LoadLibrary(libraryName);
            if (handle == IntPtr.Zero)
            {
                return false;
            }

            FreeLibrary(handle);
            return true;
        }
        catch
        {
            return false;
        }
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr LoadLibrary(string dllToLoad);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool FreeLibrary(IntPtr hModule);

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