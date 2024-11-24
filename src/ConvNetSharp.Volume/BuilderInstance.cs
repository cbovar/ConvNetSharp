using System;
using System.Numerics;
using ConvNetSharp.Volume.Double;

namespace ConvNetSharp.Volume;

public class BuilderInstance<T> where T : INumber<T>, IMinMaxValue<T>
{
    private static readonly Lazy<VolumeBuilder<T>> Singleton = new(Create);

    public static VolumeBuilder<T> Volume { get; set; } = Singleton.Value;

    public static VolumeBuilder<T> Create()
    {
        if (typeof(T) == typeof(double))
        {
            return (VolumeBuilder<T>)(object)new VolumeBuilder();
        }

        if (typeof(T) == typeof(float))
        {
            return (VolumeBuilder<T>)(object)new Single.VolumeBuilder();
        }

        throw new NotSupportedException(
            $"Volumes of type '{typeof(T).Name}' are not supported. Only Double and Single are supported.");
    }
}