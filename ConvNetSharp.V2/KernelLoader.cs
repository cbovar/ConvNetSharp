using System.Numerics;
using ManagedCuda;
using ManagedCuda.NVRTC;
using ManagedCuda.VectorTypes;

namespace ConvNetSharp.V2;

/// <summary>
///     A utility class for loading, compiling, and running CUDA kernels using the ManagedCuda library.
///     This class is specifically designed to work with GPU-based computations and integrates with the
///     ConvNetSharp framework's GpuStorage.
/// </summary>
/// <typeparam name="T">
///     The numeric type used for computations (e.g., float, double). Must implement <see cref="INumber{T}" /> and
///     <see cref="IMinMaxValue{T}" />.
/// </typeparam>
internal class KernelLoader<T> where T : struct, INumber<T>
{
    private readonly GpuContext _context;
    private readonly Dictionary<string, CudaKernel> _kernels = new();

    /// <summary>
    ///     Initializes a new instance of the <see cref="KernelLoader{T}" /> class.
    /// </summary>
    /// <param name="context">The GPU context used for managing CUDA resources.</param>
    /// <exception cref="ArgumentNullException">Thrown if <paramref name="context" /> is null.</exception>
    public KernelLoader(GpuContext context)
    {
        this._context = context ?? throw new ArgumentNullException(nameof(context));
    }

    /// <summary>
    ///     Loads and compiles a CUDA kernel from source code.
    /// </summary>
    /// <param name="name">The unique name of the kernel.</param>
    /// <param name="sourceCode">The CUDA source code to compile.</param>
    /// <exception cref="ArgumentException">Thrown if <paramref name="sourceCode" /> is null or empty.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the kernel compilation fails.</exception>
    public void LoadKernel(string name, string sourceCode)
    {
        if (string.IsNullOrWhiteSpace(sourceCode))
        {
            throw new ArgumentException("Kernel source code cannot be null or empty.", nameof(sourceCode));
        }

        var rtc = new CudaRuntimeCompiler(sourceCode, $"{name}.cu");
        try
        {
            rtc.Compile([]);
            var ptx = rtc.GetPTX();
            var kernel = this._context.CudaContext.LoadKernelFatBin(ptx, name);
            this.AddKernel(name, kernel);
        }
        catch (NVRTCException ex)
        {
            throw new InvalidOperationException($"Failed to compile kernel '{name}': {rtc.GetLogAsString()}", ex);
        }
    }


    /// <summary>
    ///     Executes a kernel with one input and one output.
    /// </summary>
    /// <param name="kernelName">The name of the kernel to execute.</param>
    /// <param name="input">The input GPU storage.</param>
    /// <param name="output">The output GPU storage.</param>
    /// <param name="extraParameters">Additional parameters to pass to the kernel.</param>
    /// <exception cref="ArgumentException">Thrown if the kernel is not found.</exception>
    public void RunKernel(string kernelName, GpuStorage<T> input, GpuStorage<T> output, params object[] extraParameters)
    {
        if (!this._kernels.TryGetValue(kernelName, out var kernel))
        {
            throw new ArgumentException($"Kernel '{kernelName}' not found.", nameof(kernelName));
        }

        this.RunKernel(kernel, input, output, extraParameters);
    }

    /// <summary>
    ///     Executes a kernel with two inputs and one output.
    /// </summary>
    /// <param name="kernelName">The name of the kernel to execute.</param>
    /// <param name="input1">The first input GPU storage.</param>
    /// <param name="input2">The second input GPU storage.</param>
    /// <param name="output">The output GPU storage.</param>
    /// <param name="extraParameters">Additional parameters to pass to the kernel.</param>
    /// <exception cref="ArgumentException">Thrown if the kernel is not found.</exception>
    public void RunKernel(string kernelName, GpuStorage<T> input1, GpuStorage<T> input2, GpuStorage<T> output,
        params object[] extraParameters)
    {
        if (!this._kernels.TryGetValue(kernelName, out var kernel))
        {
            throw new ArgumentException($"Kernel '{kernelName}' not found.", nameof(kernelName));
        }

        this.RunKernel(kernel, input1, input2, output, extraParameters);
    }

    /// <summary>
    ///     Configures and executes a kernel with one input and one output.
    /// </summary>
    /// <param name="kernel">The compiled CUDA kernel to execute.</param>
    /// <param name="input">The input GPU storage.</param>
    /// <param name="output">The output GPU storage.</param>
    /// <param name="extraParameters">Additional parameters to pass to the kernel.</param>
    private void RunKernel(CudaKernel kernel, GpuStorage<T> input, GpuStorage<T> output,
        params object[] extraParameters)
    {
        input.CopyToDevice();
        output.CopyToDevice();

        var count = (int)output.Shape.TotalLength;
        var parameters = new object[]
        {
            count,
            input.DevicePointer,
            output.DevicePointer
        };

        if (extraParameters != null)
        {
            parameters = parameters.Concat(extraParameters).ToArray();
        }

        this.ConfigureAndRunKernel(kernel, count, parameters);
    }

    /// <summary>
    ///     Configures and executes a kernel with two inputs and one output.
    /// </summary>
    /// <param name="kernel">The compiled CUDA kernel to execute.</param>
    /// <param name="input1">The first input GPU storage.</param>
    /// <param name="input2">The second input GPU storage.</param>
    /// <param name="output">The output GPU storage.</param>
    /// <param name="extraParameters">Additional parameters to pass to the kernel.</param>
    private void RunKernel(CudaKernel kernel, GpuStorage<T> input1, GpuStorage<T> input2, GpuStorage<T> output,
        params object[] extraParameters)
    {
        input1.CopyToDevice();
        input2.CopyToDevice();
        output.CopyToDevice();

        var count = (int)output.Shape.TotalLength;
        var parameters = new object[]
        {
            count,
            input1.DevicePointer,
            input2.DevicePointer,
            output.DevicePointer
        };

        if (extraParameters != null)
        {
            parameters = parameters.Concat(extraParameters).ToArray();
        }

        this.ConfigureAndRunKernel(kernel, count, parameters);
    }

    /// <summary>
    ///     Configures the thread and block dimensions for a kernel execution, and runs the kernel asynchronously.
    /// </summary>
    /// <param name="kernel">The CUDA kernel to execute.</param>
    /// <param name="count">The total number of threads required for the operation.</param>
    /// <param name="parameters">The parameters to pass to the kernel.</param>
    private void ConfigureAndRunKernel(CudaKernel kernel, int count, object[] parameters)
    {
        var threadsPerBlock = Math.Min(count, this._context.DefaultThreadsPerBlock);
        var blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

        kernel.BlockDimensions = new dim3(threadsPerBlock, 1, 1);
        kernel.GridDimensions = new dim3(blocksPerGrid, 1, 1);

        kernel.RunAsync(this._context.DefaultStream.Stream, parameters);
    }

    /// <summary>
    ///     Adds a compiled kernel to the internal dictionary.
    /// </summary>
    /// <param name="name">The unique name of the kernel.</param>
    /// <param name="kernel">The compiled CUDA kernel.</param>
    private void AddKernel(string name, CudaKernel kernel)
    {
        this._kernels[name] = kernel;
    }
}