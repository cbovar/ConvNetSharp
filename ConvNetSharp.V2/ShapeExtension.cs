using System.Numerics;

namespace ConvNetSharp.V2;

/// <summary>
///     Provides extension methods for Shape class.
/// </summary>
public static class ShapeExtension
{
    /// <summary>
    ///     Converts an object to the specified numeric type T.
    /// </summary>
    private static T ToT<T>(object o) where T : struct, INumber<T>, IMinMaxValue<T>
    {
        return (T)Convert.ChangeType(o, typeof(T));
    }

    public static Volume<T> ToVolume<T>(this Shape shape) where T : struct, INumber<T>, IMinMaxValue<T>
    {
        if (shape.Dimensions.Any(d => d <= 0))
        {
            throw new ArgumentException("Shape dimensions must be positive and non-zero.", nameof(shape));
        }

        var data = new[]
        {
            ToT<T>(shape.Dimensions[0]),
            ToT<T>(shape.Dimensions[1]),
            ToT<T>(shape.Dimensions[2]),
            ToT<T>(shape.Dimensions[3])
        };

        var volumeShape = new Shape(data.Length, 1, 1, 1); // Use a 1D shape to represent the dimensions
        return Volume<T>.Create(data, volumeShape);
    }
}