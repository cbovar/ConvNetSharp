using System;

namespace ConvNetSharp.Volume
{
    public class Layout
    {
        private readonly int[] _strides;

        public Layout(params int[] dimensions)
        {
            this._strides = new int[dimensions.Length];

            this._strides[0] = 1;
            for (var i = 1; i < dimensions.Length; i++)
            {
                this._strides[i] = this._strides[i - 1] * dimensions[i - 1];
            }
        }

        public int IndexFromCoordinates(params int[] coordinates)
        {
            var result = 0;
            var length = Math.Min(coordinates.Length, this._strides.Length);
            for (var i = 0; i < length; i++)
            {
                result += coordinates[i] * this._strides[i];
            }

            return result;
        }
    }
}