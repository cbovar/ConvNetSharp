using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;

namespace ConvNetSharp.Volume
{
    [DebuggerDisplay("Shape {PrettyPrint()}")]
    public class Shape : IEquatable<Shape>
    {
        private readonly List<int> _dimensions = new List<int>();

        public Shape()
        {
        }

        public Shape(params int[] dimensions) : this((IEnumerable<int>)dimensions)
        {

        }

        public Shape(IEnumerable<int> dimensions)
        {
            this._dimensions.AddRange(dimensions);
            UpdateTotalLength();
        }

        public Shape(Shape shape) : this(shape._dimensions.ToArray())
        {
        }

        public int DimensionCount => this._dimensions.Count;

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

            for (var i = 0; i < this.DimensionCount; i++)
            {
                if (this._dimensions[i] != other._dimensions[i])
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
            if (this._dimensions.Count <= index)
            {
                return 1;
                //throw new ArgumentException($"Shape has only {this._dimensions.Count} dimensions", nameof(index));
            }

            return this._dimensions[index];
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return ((this._dimensions?.GetHashCode() ?? 0) * 397) ^ this.TotalLength.GetHashCode();
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
                var size = this._dimensions[d];
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
            for (var i = 0; i < this._dimensions.Count - 1; i++)
            {
                sb.Append(this._dimensions[i]);
                sb.Append("x");
            }
            sb.Append(this._dimensions[this._dimensions.Count - 1]);
            return sb.ToString();
        }

        public void SetDimension(int index, int dimension)
        {
            this._dimensions[index] = dimension;
            UpdateTotalLength();
        }

        private void UpdateTotalLength()
        {
            this.TotalLength = this._dimensions.Aggregate((long)1, (acc, val) => acc * val);
        }
    }
}