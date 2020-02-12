using System;

namespace ConvNetSharp.Volume
{
    public static class ShapeExtension
    {
        private static T ToT<T>(object o) where T : struct, IEquatable<T>, IFormattable
        {
            return (T)Convert.ChangeType(o, typeof(T));
        }

        public static Volume<T> ToVolume<T>(this Shape shape) where T : struct, IEquatable<T>, IFormattable
        {
            var builder = BuilderInstance<T>.Create();
            var vol = builder.From(new[] { ToT<T>(shape.Dimensions[0]), ToT<T>(shape.Dimensions[1]), ToT<T>(shape.Dimensions[2]), ToT<T>(shape.Dimensions[3]) },
                new Shape(4));

            return vol;
        }
    }
}