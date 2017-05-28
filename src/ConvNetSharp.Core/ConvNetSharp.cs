using System;
using ConvNetSharp.Core.Ops;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core
{
    public static class ConvNetSharp<T> where T : struct, IEquatable<T>, IFormattable
    {
        public static readonly T Zero;

        public static readonly T One;

        static ConvNetSharp()
        {
            Zero = default(T);

            if (typeof(T) == typeof(double))
            {
                One = (T) (ValueType) 1.0;
            }
            else if (typeof(T) == typeof(float))
            {
                One = (T) (ValueType) 1.0f;
            }
        }

        public static Const<T> Const(Volume<T> v, string name)
        {
            return new Const<T>(v, name);
        }

        public static PlaceHolder<T> PlaceHolder(string name)
        {
            return new PlaceHolder<T>(name);
        }

        public static Variable<T> Variable(Volume<T> v, string name)
        {
            return new Variable<T>(v, name);
        }
    }
}