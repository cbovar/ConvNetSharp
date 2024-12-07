using ManagedCuda;
using ManagedCuda.CudaBlas;
using ManagedCuda.CudaDNN;

namespace ConvNetSharp.V2;

public class GpuContext : IDisposable
{
    private static readonly Lazy<GpuContext> LazyInstance = new(() => new GpuContext());

    private GpuContext()
    {
    }

    public static GpuContext Instance => LazyInstance.Value;

    public CudaContext CudaContext { get; private set; }

    public CudaDNNContext CudnnContext { get; private set; }

    public CudaBlas CublasContext { get; set; }

    public CudaStream DefaultStream { get; private set; }

    public bool IsInitialized { get; private set; }

    public void Dispose()
    {
        if (!this.IsInitialized)
        {
            return;
        }

        try
        {
            // Dispose of DefaultStream
            this.DefaultStream?.Dispose();
            this.DefaultStream = null;

            // Dispose of cuDNN context
            this.CudnnContext?.Dispose();
            this.CudnnContext = null;

            // Dispose of CUBLAS context
            this.CublasContext.Dispose();
            this.CublasContext = null;

            // Dispose of CUDA context
            this.CudaContext?.Dispose();
            this.CudaContext = null;

            Console.WriteLine("GPU context disposed.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error during GPU context disposal: {ex.Message}");
        }
        finally
        {
            this.IsInitialized = false;
        }
    }

    public void Initialize()
    {
        if (this.IsInitialized)
        {
            return;
        }

        try
        {
            // Initialize CUDA context
            this.CudaContext = new CudaContext();
            this.CudnnContext = new CudaDNNContext();
            this.CublasContext = new CudaBlas();

            // Log GPU information
            var deviceProperties = this.CudaContext.GetDeviceInfo();
            Console.WriteLine(
                $"Initialized GPU: {deviceProperties.DeviceName}, Compute Capability: {deviceProperties.ComputeCapability.Major}.{deviceProperties.ComputeCapability.Minor}");
            Console.WriteLine(
                $"Global Memory: {deviceProperties.TotalGlobalMemory / (1024 * 1024)} MB, Multiprocessors: {deviceProperties.MultiProcessorCount}");

            // Initialize a default stream
            this.DefaultStream = new CudaStream();

            this.IsInitialized = true;
        }
        catch (CudaException ex)
        {
            throw new InvalidOperationException(
                "Failed to initialize GPU context. Ensure a compatible GPU and driver are available.", ex);
        }
    }

    public void Synchronize()
    {
        if (!this.IsInitialized)
        {
            throw new InvalidOperationException("GPU context is not initialized.");
        }

        this.DefaultStream.Synchronize();
    }

    public static void Reset()
    {
        if (LazyInstance.IsValueCreated)
        {
            LazyInstance.Value.Dispose();
        }
    }
}