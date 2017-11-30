using System;

namespace ConvNetSharp.Volume
{
    /// <summary>
    /// TODO: simplify
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class VolumeBuilder<T> where T : struct, IEquatable<T>, IFormattable
    {
        public abstract Volume<T> SameAs(Shape shape);

        public abstract Volume<T> From(T[] value, Shape shape);

        public abstract Volume<T> Random(Shape shape, double mu = 0, double std = 1.0);

        public abstract Volume<T> SameAs(VolumeStorage<T> example, Shape shape);

        /// <summary>
        /// Creates a volume with given shape, using provided storage as internal storage (no copy)
        /// </summary>
        /// <param name="storage"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public abstract Volume<T> Build(VolumeStorage<T> storage, Shape shape);

        /// <summary>
        /// Creates a volume with given shape, filled with the provided value and using the same storage type as provided example
        /// </summary>
        /// <param name="example"></param>
        /// <param name="value"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public abstract Volume<T> SameAs(VolumeStorage<T> example, T value, Shape shape);
    }
}