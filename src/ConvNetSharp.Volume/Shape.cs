using System;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace ConvNetSharp.Volume
{
    [DebuggerDisplay("Shape {PrettyPrint()}")]
    public class Shape : IEquatable<Shape>
    {
        public static int None = -1; // Automatically guesses

        public static int Keep = -2; // Keep same dimension

        /// <summary>
        ///     Create shape of [1,1,c,1]
        /// </summary>
        /// <param name="c"></param>
        public Shape(int c)
        {
            if (c == 0)
            {
                throw new ArgumentException("Dimension cannot be 0", nameof(c));
            }

            this.Dimensions = new[] { 1, 1, c, 1 };
            UpdateTotalLength();
        }

        /// <summary>
        ///     Create shape of [dimensionW,dimensionH,1,1]
        /// </summary>
        /// <param name="w"></param>
        /// <param name="h"></param>
        public Shape(int w, int h)
        {
            if (w == 0)
            {
                throw new ArgumentException("Dimension cannot be 0", nameof(w));
            }

            if (h == 0)
            {
                throw new ArgumentException("Dimension cannot be 0", nameof(h));
            }

            this.Dimensions = new[] { w, h, 1, 1 };
            UpdateTotalLength();
        }

        /// <summary>
        ///     Create shape of [dimensionW,dimensionH,dimensionC,1]
        /// </summary>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <param name="c"></param>
        public Shape(int w, int h, int c)
        {
            if (w == 0)
            {
                throw new ArgumentException("Dimension cannot be 0", nameof(w));
            }

            if (h == 0)
            {
                throw new ArgumentException("Dimension cannot be 0", nameof(h));
            }

            if (c == 0)
            {
                throw new ArgumentException("Dimension cannot be 0", nameof(c));
            }

            this.Dimensions = new[] { w, h, c, 1 };
            UpdateTotalLength();
        }

        /// <summary>
        /// </summary>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <param name="c"></param>
        /// <param name="batchSize"></param>
        public Shape(int w, int h, int c, int batchSize)
        {
            if (w == 0)
            {
                throw new ArgumentException("Dimension cannot be 0", nameof(w));
            }

            if (h == 0)
            {
                throw new ArgumentException("Dimension cannot be 0", nameof(h));
            }

            if (c == 0)
            {
                throw new ArgumentException("Dimension cannot be 0", nameof(c));
            }

            if (batchSize == 0)
            {
                throw new ArgumentException("Dimension cannot be 0", nameof(batchSize));
            }

            this.Dimensions = new[] { w, h, c, batchSize };
            UpdateTotalLength();
        }

        public Shape(Shape shape)
        {
            this.Dimensions = (int[])shape.Dimensions.Clone();
            UpdateTotalLength();
        }

        public bool IsScalar { get; private set; }

        public int[] Dimensions { get; }

        public long TotalLength { get; private set; }

        public bool Equals(Shape other)
        {
            if (ReferenceEquals(null, other))
            {
                return false;
            }

            if (ReferenceEquals(this, other))
            {
                return true;
            }

            if (this.TotalLength != other.TotalLength)
            {
                return false;
            }

            for (var i = 0; i < 4; i++)
            {
                if (this.Dimensions[i] != other.Dimensions[i])
                {
                    return false;
                }
            }

            return true;
        }

        private string DimensionToString(int d)
        {
            return d == -1 ? "None" : (d == -2 ? "Keep" : d.ToString());
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj))
            {
                return false;
            }

            if (ReferenceEquals(this, obj))
            {
                return true;
            }

            if (obj.GetType() != GetType())
            {
                return false;
            }

            return Equals((Shape)obj);
        }

        public static Shape From(params int[] dimensions)
        {
            switch (dimensions.Length)
            {
                case 1: return new Shape(dimensions[0]);
                case 2: return new Shape(dimensions[0], dimensions[1]);
                case 3: return new Shape(dimensions[0], dimensions[1], dimensions[2]);
                case 4: return new Shape(dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
            }

            throw new ArgumentException($"Invalid number of dimensions {dimensions.Length}. It should be > 0 and <= 4");
        }

        public static Shape From(Shape original)
        {
            return new Shape(original);
        }

        //public int Dimensions[int index)
        //{
        //    if (this.Dimensions.Count <= index)
        //    {
        //        return 1;
        //    }

        //    if (index < 0)
        //    {
        //        index += this.DimensionCount;
        //    }

        //    if (index < 0)
        //    {
        //        index = 0;
        //    }

        //    return this.Dimensions[index];
        //}

        public override int GetHashCode()
        {
            unchecked
            {
                return ((this.Dimensions?.GetHashCode() ?? 0) * 397) ^ this.TotalLength.GetHashCode();
            }
        }

        public void GuessUnkownDimension(long totalLength)
        {
            long product = 1;
            var unknownIndex = -1;

            if (totalLength <= 0)
            {
                throw new ArgumentException($"{nameof(totalLength)} must be non-negative, not {totalLength}");
            }

            for (var d = 0; d < 4; ++d)
            {
                var size = this.Dimensions[d];
                if (size == -1)
                {
                    if (unknownIndex != -1)
                    {
                        throw new ArgumentException($"Only one input size may be - 1, not both {unknownIndex} and  {d}");
                    }

                    unknownIndex = d;
                }
                else
                {
                    if (size <= 0)
                    {
                        throw new ArgumentException($"Dimension #{d} must be non-negative, not {size}");
                    }

                    product *= size;
                }
            }

            if (unknownIndex != -1)
            {
                if (product <= 0)
                {
                    throw new ArgumentException("Reshape cannot infer the missing input size " +
                                                "for an empty volume unless all specified " +
                                                "input sizes are non-zero");
                }

                var missing = totalLength / product;

                if (missing * product != totalLength)
                {
                    throw new ArgumentException($"Input to reshape is a tensor with totalLength={totalLength}, " +
                                                $"but the requested shape requires totalLength to be a multiple of {product}");
                }

                SetDimension(unknownIndex, (int)missing);
            }
            else
            {
                if (product != totalLength)
                {
                    throw new ArgumentException("incompatible dimensions provided");
                }
            }
        }

        public string PrettyPrint(string sep = "x")
        {
            var sb = new StringBuilder();
            for (var i = 0; i < 3; i++)
            {
                sb.Append(DimensionToString(this.Dimensions[i]));
                sb.Append(sep);
            }

            sb.Append(DimensionToString(this.Dimensions[3]));
            return sb.ToString();
        }

        public void SetDimension(int index, int dimension)
        {
            if (index < 0)
            {
                throw new ArgumentException("index cannot be negative", nameof(index));
            }

            this.Dimensions[index] = dimension;
            UpdateTotalLength();
        }

        public override string ToString()
        {
            return PrettyPrint();
        }

        private void UpdateTotalLength()
        {
            this.TotalLength = this.Dimensions.Aggregate((long)1, (acc, val) => acc * val);
            this.IsScalar = this.Dimensions[0] == this.Dimensions[1] == (this.Dimensions[2] == 1);
        }
    }
}