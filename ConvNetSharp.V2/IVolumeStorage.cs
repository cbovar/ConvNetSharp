namespace ConvNetSharp.V2;

/// <summary>
///     Interface representing the storage layer for a volume, defining methods for accessing and modifying data,
///     performing operations, and handling multi-dimensional tensor-like structures.
/// </summary>
/// <typeparam name="T">The numeric type (e.g., float, double).</typeparam>
public interface IVolumeStorage<T> where T : struct
{
    /// <summary>
    ///     Gets the shape of the volume, representing its dimensions.
    /// </summary>
    Shape Shape { get; }

    /// <summary>
    ///     Gets the device type where the data is stored (e.g., CPU or GPU).
    /// </summary>
    Device Device { get; }

    /// <summary>
    ///     Retrieves the entire data array of the volume in its flattened form.
    /// </summary>
    /// <returns>An array containing all elements in the volume.</returns>
    T[] GetData();

    /// <summary>
    ///     Sets the entire data array of the volume.
    /// </summary>
    /// <param name="data">An array containing the data to be set.</param>
    /// <exception cref="ArgumentException">
    ///     Thrown when the length of the provided data does not match the total length of the volume.
    /// </exception>
    void SetData(T[] data);

    /// <summary>
    ///     Retrieves the value at the specified n-dimensional coordinates.
    /// </summary>
    /// <param name="coordinates">An array of integers representing the coordinates.</param>
    /// <returns>The value at the specified coordinates.</returns>
    T Get(int[] coordinates);

    /// <summary>
    ///     Retrieves the value at the specified 4D coordinates.
    /// </summary>
    /// <param name="w">The width coordinate.</param>
    /// <param name="h">The height coordinate.</param>
    /// <param name="c">The channel coordinate.</param>
    /// <param name="n">The batch coordinate.</param>
    /// <returns>The value at the specified 4D coordinates.</returns>
    T Get(int w, int h, int c, int n);

    /// <summary>
    ///     Retrieves the value at the specified 3D coordinates.
    /// </summary>
    /// <param name="w">The width coordinate.</param>
    /// <param name="h">The height coordinate.</param>
    /// <param name="c">The channel coordinate.</param>
    /// <returns>The value at the specified 3D coordinates.</returns>
    T Get(int w, int h, int c);

    /// <summary>
    ///     Retrieves the value at the specified 2D coordinates.
    /// </summary>
    /// <param name="w">The width coordinate.</param>
    /// <param name="h">The height coordinate.</param>
    /// <returns>The value at the specified 2D coordinates.</returns>
    T Get(int w, int h);

    /// <summary>
    ///     Retrieves the value at the specified flat index.
    /// </summary>
    /// <param name="i">The flat index.</param>
    /// <returns>The value at the specified index.</returns>
    T Get(int i);

    /// <summary>
    ///     Sets the value at the specified n-dimensional coordinates.
    /// </summary>
    /// <param name="coordinates">An array of integers representing the coordinates.</param>
    /// <param name="value">The value to set at the specified coordinates.</param>
    void Set(int[] coordinates, T value);

    /// <summary>
    ///     Sets the value at the specified 4D coordinates.
    /// </summary>
    /// <param name="w">The width coordinate.</param>
    /// <param name="h">The height coordinate.</param>
    /// <param name="c">The channel coordinate.</param>
    /// <param name="n">The batch coordinate.</param>
    /// <param name="value">The value to set.</param>
    void Set(int w, int h, int c, int n, T value);

    /// <summary>
    ///     Sets the value at the specified 3D coordinates.
    /// </summary>
    /// <param name="w">The width coordinate.</param>
    /// <param name="h">The height coordinate.</param>
    /// <param name="c">The channel coordinate.</param>
    /// <param name="value">The value to set.</param>
    void Set(int w, int h, int c, T value);

    /// <summary>
    ///     Sets the value at the specified 2D coordinates.
    /// </summary>
    /// <param name="w">The width coordinate.</param>
    /// <param name="h">The height coordinate.</param>
    /// <param name="value">The value to set.</param>
    void Set(int w, int h, T value);

    /// <summary>
    ///     Sets the value at the specified flat index.
    /// </summary>
    /// <param name="i">The flat index.</param>
    /// <param name="value">The value to set at the specified index.</param>
    void Set(int i, T value);

    /// <summary>
    ///     Applies an element-wise operation to all elements in the volume.
    /// </summary>
    /// <param name="operation">The function to apply to each element.</param>
    void ApplyOperation(Func<T, T> operation);

    /// <summary>
    ///     Adds the corresponding elements of this volume and another volume, storing the result in a third volume.
    /// </summary>
    /// <param name="other">The other volume to add.</param>
    /// <param name="result">The volume to store the result.</param>
    void Add(IVolumeStorage<T> other, IVolumeStorage<T> result);

    /// <summary>
    ///     Accumulates the data from the current storage into another storage in-place.
    /// </summary>
    /// <param name="result">The storage where the data is accumulated.</param>
    void Add(IVolumeStorage<T> result);

    /// <summary>
    ///     Clears the volume by setting all its elements to zero.
    /// </summary>
    void Clear();

    /// <summary>
    ///     Performs a convolution operation on the input volume using the specified filter.
    /// </summary>
    /// <param name="filterStorage">The filter volume to use for convolution.</param>
    /// <param name="xpad">The padding size along the width dimension.</param>
    /// <param name="ypad">The padding size along the height dimension.</param>
    /// <param name="stride">The stride size for the convolution.</param>
    /// <param name="outputStorage">The volume to store the convolution result.</param>
    void Convolution(IVolumeStorage<T> filterStorage, int xpad, int ypad, int stride, IVolumeStorage<T> outputStorage);

    /// <summary>
    ///     Repeats the elements of this volume according to the specified replication factors,
    ///     and stores the result in the provided result volume.
    /// </summary>
    /// <param name="reps">The replication factors volume.</param>
    /// <param name="result">The volume to store the tiled output.</param>
    void Tile(IVolumeStorage<T> reps, IVolumeStorage<T> result);

    /// <summary>
    ///     Multiplies all elements of this volume by a scalar value, optionally in-place.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply by.</param>
    /// <param name="inPlace">Whether to perform the operation in-place. Defaults to false.</param>
    /// <returns>A new volume if inPlace is false; otherwise, the updated volume.</returns>
    IVolumeStorage<T> MultiplyByScalar(T scalar, bool inPlace = false);

    /// <summary>
    ///     Subtracts the elements of another volume from this volume, storing the result in a third volume.
    /// </summary>
    /// <param name="rightStorage">The volume to subtract from this volume.</param>
    /// <param name="resultStorage">The volume to store the result.</param>
    void Subtract(IVolumeStorage<T> rightStorage, IVolumeStorage<T> resultStorage);

    /// <summary>
    ///     Performs matrix multiplication between two volumes and stores the result in a third volume.
    /// </summary>
    /// <param name="rightStorage">The right-hand operand of the matrix multiplication.</param>
    /// <param name="resultStorage">The storage where the result of the matrix multiplication will be stored.</param>
    /// <exception cref="ArgumentException">
    ///     Thrown when the dimensions of the volumes are incompatible for matrix
    ///     multiplication.
    /// </exception>
    public void MatMultiply(IVolumeStorage<T> rightStorage, IVolumeStorage<T> resultStorage);

    /// <summary>
    ///     Performs element-wise multiplication between the current volume storage and another volume storage,
    ///     and stores the result in the provided result storage.
    /// </summary>
    /// <param name="rightStorage">The right-hand operand volume storage for the element-wise multiplication.</param>
    /// <param name="resultStorage">The volume storage where the result of the element-wise multiplication will be stored.</param>
    /// <exception cref="ArgumentNullException">
    ///     Thrown when <paramref name="rightStorage" /> or
    ///     <paramref name="resultStorage" /> is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    ///     Thrown when the shapes of the volume storages are not compatible for element-wise
    ///     multiplication.
    /// </exception>
    /// <remarks>
    ///     The method computes the operation:
    ///     <c>result[i] = this[i] * rightStorage[i]</c>
    ///     for each element in the volumes. The shapes of the volumes must match or be broadcastable for the operation to
    ///     succeed.
    /// </remarks>
    void Multiply(IVolumeStorage<T> rightStorage, IVolumeStorage<T> resultStorage);
}