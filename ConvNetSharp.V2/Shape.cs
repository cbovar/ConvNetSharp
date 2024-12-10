namespace ConvNetSharp.V2;

/// <summary>
///     Represents the shape of a tensor or multidimensional array with up to four dimensions:
///     width (w), height (h), channels (c), and batch size. This class provides utilities for validating,
///     inferring, and manipulating tensor dimensions, as well as for checking common cases like
///     scalars, vectors, and matrices.
/// </summary>
public class Shape : IEquatable<Shape>
{
    public const int None = -1;
    public const int Keep = -2;

    /// <summary>
    ///     Creates a default shape [1, 1, 1, 1].
    /// </summary>
    public Shape() : this(1, 1, 1, 1)
    {
    }

    /// <summary>
    ///     Creates a shape with channels (c) specified: [1 (width), 1 (height), c, 1 (batch size)].
    /// </summary>
    public Shape(int c) : this(new[] { 1, 1, c, 1 })
    {
    }

    /// <summary>
    ///     Creates a shape with width (w) and height (h) specified: [w, h, 1 (channels), 1 (batch size)].
    /// </summary>
    public Shape(int w, int h) : this(new[] { w, h, 1, 1 })
    {
    }

    /// <summary>
    ///     Creates a shape with width (w), height (h), and channels (c) specified: [w, h, c, 1 (batch size)].
    /// </summary>
    public Shape(int w, int h, int c) : this(new[] { w, h, c, 1 })
    {
    }

    /// <summary>
    ///     Creates a shape with all four dimensions specified: [w (width), h (height), c (channels), batchSize].
    /// </summary>
    public Shape(int w, int h, int c, int batchSize) : this(new[] { w, h, c, batchSize })
    {
    }

    /// <summary>
    ///     Creates a shape from an array of dimensions [w, h, c, batchSize].
    /// </summary>
    public Shape(int[] dimensions)
    {
        if (dimensions is not { Length: 4 })
        {
            throw new ArgumentException("Dimensions array must contain exactly 4 elements.");
        }

        if (dimensions.Any(d => d == 0))
        {
            throw new ArgumentException("Dimension cannot be 0.");
        }

        this.Dimensions = dimensions;
        this.UpdateTotalLength();
    }

    /// <summary>
    ///     Creates a copy of an existing shape.
    /// </summary>
    public Shape(Shape shape)
    {
        this.Dimensions = (int[])shape.Dimensions.Clone();
        this.UpdateTotalLength();
    }

    /// <summary>
    ///     Dimensions of the shape in the order: [w (width), h (height), c (channels), batchSize].
    /// </summary>
    public int[] Dimensions { get; }

    /// <summary>
    ///     Total number of elements in the shape.
    /// </summary>
    public long TotalLength { get; private set; }

    /// <summary>
    ///     Whether the shape represents a scalar (1x1x1x1).
    /// </summary>
    public bool IsScalar { get; private set; }

    /// <summary>
    ///     Whether the shape represents a vector (1D array).
    ///     Vector = [length, 1, 1, 1].
    /// </summary>
    public bool IsVector => this.Dimensions[1] == 1 && this.Dimensions[2] == 1 && this.Dimensions[3] == 1;

    /// <summary>
    ///     Whether the shape represents a matrix (2D array).
    ///     Matrix = [width, height, 1, 1].
    /// </summary>
    public bool IsMatrix => this.Dimensions[2] == 1 && this.Dimensions[3] == 1;

    public bool Equals(Shape other)
    {
        if (other == null || this.TotalLength != other.TotalLength)
        {
            return false;
        }

        return this.Dimensions.SequenceEqual(other.Dimensions);
    }

    /// <summary>
    ///     Updates the total length and scalar property based on current dimensions.
    /// </summary>
    private void UpdateTotalLength()
    {
        this.TotalLength = this.Dimensions.Aggregate(1L, (a, b) => a * b);
        this.IsScalar = this.Dimensions[0] == 1 && this.Dimensions[1] == 1 && this.Dimensions[2] == 1 &&
                        this.Dimensions[3] == 1;
    }

    /// <summary>
    ///     Returns a string representation of the shape, formatted as a JSON array or with a custom separator.
    /// </summary>
    public string PrettyPrint(string sep = "x", bool asJson = false)
    {
        if (asJson)
        {
            return $"[{string.Join(", ", this.Dimensions)}]";
        }

        return string.Join(sep, this.Dimensions.Select(DimensionToString));
    }

    private static string DimensionToString(int d)
    {
        return d switch
        {
            None => "None",
            Keep => "Keep",
            _ => d.ToString()
        };
    }


    /// <summary>
    ///     Infers unknown (-1) dimensions based on the total length.
    /// </summary>
    public void GuessUnknownDimension(long totalLength)
    {
        long product = 1;
        var unknownIndex = -1;

        if (totalLength <= 0)
        {
            throw new ArgumentException($"{nameof(totalLength)} must be positive.");
        }

        for (var d = 0; d < 4; ++d)
        {
            var size = this.Dimensions[d];
            if (size == -1)
            {
                if (unknownIndex != -1)
                {
                    throw new ArgumentException("Only one dimension can be -1, not multiple.");
                }

                unknownIndex = d;
            }
            else
            {
                if (size <= 0)
                {
                    throw new ArgumentException($"Dimension #{d} must be positive.");
                }

                product *= size;
            }
        }

        if (unknownIndex != -1)
        {
            var missing = totalLength / product;

            if (missing * product != totalLength)
            {
                throw new ArgumentException(
                    $"Total length {totalLength} is not compatible with the specified dimensions.");
            }

            this.SetDimension(unknownIndex, (int)missing);
        }
        else if (product != totalLength)
        {
            throw new ArgumentException("Specified dimensions are incompatible with the total length.");
        }
    }

    /// <summary>
    ///     Sets a specific dimension and updates the shape.
    /// </summary>
    public void SetDimension(int index, int dimension)
    {
        if (index < 0 || index >= this.Dimensions.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(index), "Index must be between 0 and 3.");
        }

        this.Dimensions[index] = dimension;
        this.UpdateTotalLength();
    }

    public override string ToString()
    {
        return this.PrettyPrint();
    }

    public override bool Equals(object obj)
    {
        return obj is Shape other && this.Equals(other);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(this.Dimensions, this.TotalLength);
    }

    /// <summary>
    ///     Creates a shape from an arbitrary number of dimensions.
    /// </summary>
    public static Shape From(params int[] dimensions)
    {
        if (dimensions.Any(dim => dim == 0))
        {
            throw new ArgumentException("Dimensions cannot contain 0.");
        }

        if (dimensions.Length is < 1 or > 4)
        {
            throw new ArgumentException("Dimensions must have between 1 and 4 elements.");
        }

        var fullDimensions = new[] { 1, 1, 1, 1 };
        Array.Copy(dimensions, fullDimensions, dimensions.Length);

        return new Shape(fullDimensions);
    }

    public static Shape From(Shape original)
    {
        return new Shape(original);
    }

    /// <summary>
    ///     Determines whether the current shape can be broadcasted to the target shape.
    /// </summary>
    /// <param name="current">The current shape.</param>
    /// <param name="target">The target shape.</param>
    /// <returns>True if the current shape can be broadcasted to the target shape; otherwise, false.</returns>
    public bool IsBroadcastableTo(Shape target)
    {
        if (target == null)
        {
            throw new ArgumentNullException("Shapes cannot be null.");
        }

        var currentDims = this.Dimensions;
        var targetDims = target.Dimensions;

        // Ensure that the number of dimensions match by left-padding with ones
        var currentPadded = Enumerable.Repeat(1, Math.Max(0, targetDims.Length - currentDims.Length))
            .Concat(currentDims)
            .ToArray();
        var targetPadded = Enumerable.Repeat(1, Math.Max(0, currentDims.Length - targetDims.Length))
            .Concat(targetDims)
            .ToArray();

        // Check broadcasting compatibility
        for (var i = 0; i < currentPadded.Length; i++)
        {
            if (currentPadded[i] != targetPadded[i] && currentPadded[i] != 1)
            {
                return false; // Dimensions must match or one of them must be 1
            }
        }

        return true;
    }

    /// <summary>
    ///     Computes the strides for a given shape. Strides represent the step size in memory for each dimension.
    ///     For example, the stride for the last dimension is always 1, and the stride for the second-to-last dimension
    ///     is the size of the last dimension, and so on.
    /// </summary>
    /// <returns>An array of strides corresponding to the shape's dimensions.</returns>
    public int[] GetStrides()
    {
        var dimensions = this.Dimensions;
        var strides = new int[dimensions.Length];

        strides[^1] = 1; // Last dimension stride is 1.
        for (var i = dimensions.Length - 2; i >= 0; i--)
        {
            strides[i] = strides[i + 1] * dimensions[i + 1];
        }

        return strides;
    }


    /// <summary>
    ///     Calculates the resulting shape when two shapes are broadcasted together.
    ///     Broadcasting allows arrays of different shapes to be used together by expanding smaller dimensions.
    ///     For example, [1, 3] and [4, 3] can broadcast to [4, 3].
    /// </summary>
    /// <param name="shape1">The first shape.</param>
    /// <param name="shape2">The second shape.</param>
    /// <returns>The resulting broadcasted shape.</returns>
    /// <exception cref="ArgumentException">Thrown if the shapes are not broadcastable.</exception>
    public static Shape GetBroadcastedShape(Shape shape1, Shape shape2)
    {
        var dims1 = shape1.Dimensions;
        var dims2 = shape2.Dimensions;

        var maxLength = Math.Max(dims1.Length, dims2.Length);
        var broadcastedDims = new int[maxLength];

        for (var i = 0; i < maxLength; i++)
        {
            var dim1 = i < dims1.Length ? dims1[dims1.Length - 1 - i] : 1;
            var dim2 = i < dims2.Length ? dims2[dims2.Length - 1 - i] : 1;

            if (dim1 != 1 && dim2 != 1 && dim1 != dim2)
            {
                throw new ArgumentException("Shapes are not broadcastable.");
            }

            broadcastedDims[maxLength - 1 - i] = Math.Max(dim1, dim2);
        }

        return new Shape(broadcastedDims);
    }

    /// <summary>
    ///     Converts a flat index into an n-dimensional coordinate based on the shape's strides.
    ///     This is useful when working with linearized data (like a flat array) that needs to map back to
    ///     multidimensional coordinates.
    /// </summary>
    /// <param name="flatIndex">The flat index to convert.</param>
    /// <param name="strides">The strides corresponding to the shape.</param>
    /// <returns>An array of n-dimensional coordinates.</returns>
    public int[] GetCoordinates(int flatIndex, int[] strides)
    {
        var dimensions = this.Dimensions;
        var coordinates = new int[dimensions.Length];

        for (var i = 0; i < dimensions.Length; i++)
        {
            coordinates[i] = flatIndex / strides[i];
            flatIndex %= strides[i];
        }

        return coordinates;
    }

    /// <summary>
    ///     Adjusts n-dimensional coordinates to account for broadcasting.
    ///     If a dimension of the current shape is 1, the coordinate for that dimension is always 0 (broadcasted).
    ///     For other dimensions, the corresponding coordinate from targetCoordinates is used.
    /// </summary>
    /// <param name="targetCoordinates">The target coordinates in the broadcasted shape.</param>
    /// <param name="strides">The strides corresponding to the broadcasted shape.</param>
    /// <returns>An array of adjusted coordinates for broadcasting.</returns>
    public int[] GetBroadcastCoordinates(int[] targetCoordinates, int[] strides)
    {
        if (targetCoordinates.Length != this.Dimensions.Length)
        {
            throw new ArgumentException("Target coordinates and shape dimensions must have the same length.");
        }

        var broadcastCoordinates = new int[this.Dimensions.Length];

        for (var i = 0; i < this.Dimensions.Length; i++)
        {
            if (this.Dimensions[i] == 1)
            {
                // Dimension is broadcasted, so always use 0 as the coordinate
                broadcastCoordinates[i] = 0;
            }
            else
            {
                // Otherwise, use the target coordinate
                broadcastCoordinates[i] = targetCoordinates[i];
            }
        }

        return broadcastCoordinates;
    }


    /// <summary>
    ///     Converts n-dimensional coordinates into a flat index. This is useful for accessing
    ///     elements in a flat array that represents a multidimensional tensor.
    /// </summary>
    /// <param name="coordinates">An array of n-dimensional coordinates.</param>
    /// <returns>The corresponding flat index.</returns>
    public int GetFlatIndex(int[] coordinates)
    {
        if (coordinates.Length != this.Dimensions.Length)
        {
            throw new ArgumentException("Coordinate dimensions do not match the shape.");
        }

        var strides = this.GetStrides();
        var flatIndex = 0;

        for (var i = 0; i < this.Dimensions.Length; i++)
        {
            // Validate coordinate
            if (coordinates[i] < 0 || coordinates[i] >= this.Dimensions[i])
            {
                throw new IndexOutOfRangeException(
                    $"Coordinate at index {i} is out of bounds for dimension size {this.Dimensions[i]}.");
            }

            flatIndex += coordinates[i] * strides[i];
        }

        return flatIndex;
    }
}