using System.Numerics;

namespace ConvNetSharp.V2;

public static class MathHelpers
{
    public static T Log<T>(T value) where T : struct, INumber<T>
    {
        if (value < T.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(value), "Logarithm is not defined for negative values.");
        }

        if (typeof(T) == typeof(float))
        {
            return (T)(object)MathF.Log((float)(object)value);
        }

        if (typeof(T) == typeof(double))
        {
            return (T)(object)Math.Log((double)(object)value);
        }

        throw new NotSupportedException($"Logarithm is not supported for type {typeof(T)}");
    }
}