using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace ConvNetSharp.Volume
{
    [DebuggerDisplay("Shape {PrettyPrint()}")]
    public class Shape : IEquatable<Shape>
    {
        public Shape()
        {
        }

        public Shape(params int[] dimensions) : this((IEnumerable<int>)dimensions)
        {
        }

        public Shape(IEnumerable<int> dimensions)
        {
            this.Dimensions = dimensions.ToArray();
            UpdateTotalLength();
        }

        public Shape(Shape shape) : this(shape.Dimensions.ToArray())
        {
        }

        public int[] Dimensions { get; }

        public int DimensionCount => this.Dimensions.Length;

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

            if (this.Dimensions.Length != other.Dimensions.Length)
            {
                return false;
            }

            for (var i = 0; i < this.DimensionCount; i++)
            {
                if (this.Dimensions[i] != other.Dimensions[i])
                {
                    return false;
                }
            }

            return true;
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

        public int GetDimension(int index)
        {
            if (this.Dimensions.Length <= index)
            {
                return 1;
            }

            return this.Dimensions[index];
        }

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
            var numDims = this.DimensionCount;

            if (totalLength <= 0)
            {
                throw new ArgumentException($"{nameof(totalLength)} must be non-negative, not {totalLength}");
            }

            for (var d = 0; d < numDims; ++d)
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
                    throw new ArgumentException($"Input to reshape is a tensor with {totalLength} values, " +
                                                $"but the requested shape requires a multiple of {product}");
                }

                SetDimension(unknownIndex, (int)missing);
            }
            else
            {
                if (product != totalLength)
                {
                    throw new ArgumentException("Imcompatible dimensions provided");
                }
            }
        }

        public string PrettyPrint()
        {
            var sb = new StringBuilder();
            for (var i = 0; i < this.Dimensions.Length - 1; i++)
            {
                sb.Append(this.Dimensions[i]);
                sb.Append("x");
            }
            sb.Append(this.Dimensions[this.Dimensions.Length - 1]);
            return sb.ToString();
        }

        public void SetDimension(int index, int dimension)
        {
            this.Dimensions[index] = dimension;
            UpdateTotalLength();
        }

        private void UpdateTotalLength()
        {
            this.TotalLength = this.Dimensions.Aggregate((long)1, (acc, val) => acc * val);
        }
    }
}