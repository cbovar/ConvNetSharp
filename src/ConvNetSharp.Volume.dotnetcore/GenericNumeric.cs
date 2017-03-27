using System;

namespace ConvNetSharp.Volume
{
    public static class GenericNumeric<T> where T : struct, IEquatable<T>
    {
        public static T One()
        {
            if (typeof(T) == typeof(double))
            {
                return (T) (ValueType) 1.0;
            }

            throw new NotImplementedException();
        }
    }
}