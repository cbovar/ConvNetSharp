using System;

namespace ConvNetSharp.Volume
{
    public class BuilderInstance<T> where T : struct, IEquatable<T>, IFormattable
    {
        private static readonly Lazy<VolumeBuilder<T>> Singleton = new Lazy<VolumeBuilder<T>>(Create);

        public static VolumeBuilder<T> Volume { get; set; } = Singleton.Value;

        public static VolumeBuilder<T> Create()
        {
            if (typeof(T) == typeof(double))
            {
                return (VolumeBuilder<T>)(object)new Double.VolumeBuilder();
            }
            if (typeof(T) == typeof(float))
            {
                return (VolumeBuilder<T>)(object)new Single.VolumeBuilder();
            }

            throw new NotSupportedException(
                $"Volumes of type '{typeof(T).Name}' are not supported. Only Double and Single are supported.");
        }
    }
}