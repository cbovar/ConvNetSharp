using System;
using System.Diagnostics;
using System.Text;

namespace ConvNetSharp.Volume
{
    [DebuggerDisplay("Volume {Shape.PrettyPrint()}")]
    public abstract class Volume<T>
        where T : struct, IEquatable<T>, IFormattable
    {
        protected Volume(VolumeStorage<T> storage)
        {
            this.Storage = storage;
        }

        public VolumeStorage<T> Storage { get; }

        public Shape Shape => this.Storage.Shape;

        public Volume<T> Add(Volume<T> other)
        {
            var sameChannels = other.Shape.GetDimension(2) == this.Shape.GetDimension(2);
            if (!Equals(other.Shape, this.Shape) && !sameChannels)
            {
                throw new ArgumentException("Both volume should have the same shape.");
            }

            if (sameChannels)
            {
            }

            var result = BuilderInstance<T>.Volume.SameAs(this.Storage, this.Shape);
            DoAdd(other, result);
            return result;
        }

        public void BiasGradient(Volume<T> biasGradient)
        {
            DoBiasGradient(biasGradient);
        }

        public void Clear()
        {
            this.Storage.Clear();
        }

        public Volume<T> Clone()
        {
            var data = new T[this.Shape.TotalLength];
            Array.Copy(ToArray(), data, data.Length);

            return BuilderInstance<T>.Volume.SameAs(data, this.Shape);
        }

        public Volume<T> Convolve(Volume<T> filters, int pad, int stride)
        {
            var outputDepth = filters.Shape.GetDimension(3);
            var outputWidth =
                (int)
                Math.Floor((this.Shape.GetDimension(0) + pad * 2 - filters.Shape.GetDimension(0)) / (double)stride + 1);
            var outputHeight =
                (int)
                Math.Floor((this.Shape.GetDimension(1) + pad * 2 - filters.Shape.GetDimension(1)) / (double)stride + 1);

            var result = BuilderInstance<T>.Volume.SameAs(this.Storage,
                new Shape(outputWidth, outputHeight, outputDepth, this.Shape.GetDimension(3)));
            DoConvolution(filters, pad, stride, result);
            return result;
        }

        public void ConvolveGradient(Volume<T> filters, Volume<T> outputGradients, Volume<T> inputGradient,
            Volume<T> filterGradient, int pad, int stride)
        {
            DoConvolutionGradient(filters, outputGradients, inputGradient, filterGradient, pad, stride);
        }

        public abstract void DoAdd(Volume<T> other, Volume<T> result);

        protected abstract void DoBiasGradient(Volume<T> biasGradient);

        public abstract void DoConvolution(Volume<T> filters, int pad, int stride, Volume<T> result);

        protected abstract void DoConvolutionGradient(Volume<T> filters, Volume<T> outputGradients,
            Volume<T> inputGradient,
            Volume<T> filterGradient, int pad, int stride);

        public abstract void DoMultiply(Volume<T> result, T factor);

        public abstract void DoMultiply(Volume<T> right, Volume<T> result);

        public abstract void DoSubtractFrom(Volume<T> other, Volume<T> result);

        public abstract void DoNegate(Volume<T> result);

        public abstract void DoPool(Volume<T> result, int windowWidth, int windowHeight,
            int horizontalPad, int verticalPad, int horizontalStride, int verticalStride);

        public abstract void DoPoolGradient(Volume<T> input, Volume<T> outputGradient,
            Volume<T> inputGradient, int windowWidth, int windowHeight,
            int horizontalPad, int verticalPad, int horizontalStride, int verticalStride);

        public abstract void DoRelu(Volume<T> result);

        public abstract void DoReluGradient(Volume<T> input, Volume<T> outputGradient, Volume<T> inputGradient);

        public abstract void DoSigmoid(Volume<T> result);

        public abstract void DoSigmoidGradient(Volume<T> input, Volume<T> outputGradient, Volume<T> inputGradient);

        public abstract void DoSoftMax(Volume<T> result);

        public abstract void DoSoftMaxGradient(Volume<T> outputGradient, Volume<T> inputGradient);
        
        public abstract void DoTanh(Volume<T> result);

        public abstract void DoTanhGradient(Volume<T> input, Volume<T> outputGradient, Volume<T> inputGradient);

        public T Get(int[] coordinates)
        {
            return this.Storage.Get(coordinates);
        }

        public T Get(int w, int h, int c, int n)
        {
            return this.Storage.Get(w, h, c, n);
        }

        public T Get(int w, int h, int c)
        {
            //Debug.Assert(this.Shape.DimensionCount == 3, "Shape should have 3 dimensions");
            return this.Storage.Get(w, h, c, 0);
        }

        public T Get(int w, int h)
        {
            // Debug.Assert(this.Shape.DimensionCount == 2, "Shape should have 2 dimensions");
            return this.Storage.Get(w, h, 0, 0);
        }

        public T Get(int i)
        {
            // Debug.Assert(this.Shape.DimensionCount == 1, "Shape should have 1 dimension");
            return this.Storage.Get(i);
        }

        public void MapInplace(Func<T, T> f)
        {
            this.Storage.MapInplace(f);
        }

        public void MapInplace(Func<T, T, T> f, Volume<T> other)
        {
            this.Storage.MapInplace(f, other.Storage);
        }

        public Volume<T> Multiply(T factor)
        {
            var result = BuilderInstance<T>.Volume.SameAs(this.Storage, this.Shape);
            DoMultiply(result, factor);
            return result;
        }

        public Volume<T> Negate()
        {
            var result = BuilderInstance<T>.Volume.SameAs(this.Storage, this.Shape);
            DoNegate(result);
            return result;
        }

        public static Volume<T> operator +(Volume<T> leftSide, Volume<T> rightSide)
        {
            return leftSide.Add(rightSide);
        }

        public static Volume<T> operator *(Volume<T> volume, T factor)
        {
            return volume.Multiply(factor);
        }

        public static Volume<T> operator -(Volume<T> leftSide, Volume<T> rightSide)
        {
            return rightSide.SubtractFrom(leftSide);
        }

        public static Volume<T> operator -(Volume<T> volume)
        {
            return volume.Negate();
        }

        public Volume<T> Pool(int windowWidth, int windowHeight, int pad, int stride)
        {
            return Pool(windowWidth, windowHeight, pad, pad, stride, stride);
        }

        public Volume<T> Pool(int windowWidth, int windowHeight, int horizontalPad, int verticalPad,
            int horizontalStride,
            int verticalStride)
        {
            var outputN = this.Shape.GetDimension(3);
            var outputDepth = this.Shape.GetDimension(2);
            var outputWidth =
                (int)
                Math.Floor((this.Shape.GetDimension(0) + horizontalPad * 2 - windowWidth) / (double)horizontalStride +
                           1);
            var outputHeight =
                (int)
                Math.Floor((this.Shape.GetDimension(1) + verticalPad * 2 - windowHeight) / (double)verticalStride + 1);

            var result = BuilderInstance<T>.Volume.SameAs(this.Storage,
                new Shape(outputWidth, outputHeight, outputDepth, outputN));
            DoPool(result, windowWidth, windowHeight, horizontalPad, verticalPad, horizontalStride, verticalStride);
            return result;
        }

        public Volume<T> PoolGradient(Volume<T> input, Volume<T> outputGradient, int windowWidth, int windowHeight,
            int pad, int stride)
        {
            return PoolGradient(input, outputGradient, windowWidth, windowHeight, pad, pad, stride, stride);
        }

        public Volume<T> PoolGradient(Volume<T> input, Volume<T> outputGradient, int windowWidth, int windowHeight,
            int horizontalPad, int verticalPad, int horizontalStride,
            int verticalStride)
        {
            var inputGradient = BuilderInstance<T>.Volume.SameAs(this.Storage, input.Shape);
            DoPoolGradient(input, outputGradient, inputGradient, windowWidth, windowHeight, horizontalPad, verticalPad,
                horizontalStride, verticalStride);
            return inputGradient;
        }

        public Volume<T> Relu()
        {
            var result = BuilderInstance<T>.Volume.SameAs(this.Storage, this.Shape);
            DoRelu(result);
            return result;
        }

        public Volume<T> ReluGradient(Volume<T> input, Volume<T> outputGradient)
        {
            var inputGradient = BuilderInstance<T>.Volume.SameAs(this.Storage, this.Shape);
            DoReluGradient(input, outputGradient, inputGradient);
            return inputGradient;
        }

        public Volume<T> ReShape(params int[] dimensions)
        {
            var shape = new Shape(dimensions);
            shape.GuessUnkownDimension(this.Shape.TotalLength);

            return BuilderInstance<T>.Volume.Build(this.Storage, shape);
        }

        public void Set(int[] coordinates, T value)
        {
            this.Storage.Set(coordinates, value);
        }

        public void Set(int w, int h, int c, int n, T value)
        {
            this.Storage.Set(w, h, c, n, value);
        }

        public void Set(int w, int h, int c, T value)
        {
            this.Storage.Set(w, h, c, 0, value);
        }

        public void Set(int w, int h, T value)
        {
            this.Storage.Set(w, h, 0, 0, value);
        }

        public void Set(int i, T value)
        {
            this.Storage.Set(i, value);
        }

        public Volume<T> Sigmoid()
        {
            var result = BuilderInstance<T>.Volume.SameAs(this.Storage, this.Shape);
            DoSigmoid(result);
            return result;
        }

        public Volume<T> SigmoidGradient(Volume<T> input, Volume<T> outputGradient)
        {
            var inputGradient = BuilderInstance<T>.Volume.SameAs(this.Storage, this.Shape);
            DoSigmoidGradient(input, outputGradient, inputGradient);
            return inputGradient;
        }

        public Volume<T> SoftMax()
        {
            var result = BuilderInstance<T>.Volume.SameAs(this.Storage, this.Shape);
            DoSoftMax(result);
            return result;
        }

        public Volume<T> SoftMaxGradient(Volume<T> outputGradient)
        {
            var inputGradient = BuilderInstance<T>.Volume.SameAs(this.Storage, this.Shape);
            DoSoftMaxGradient(outputGradient, inputGradient);
            return inputGradient;
        }

        public Volume<T> SubtractFrom(Volume<T> other)
        {
            if (!Equals(other.Shape, this.Shape))
            {
                throw new ArgumentException("Both volume should have the same shape.");
            }

            var result = BuilderInstance<T>.Volume.SameAs(this.Storage, this.Shape);
            DoSubtractFrom(other, result);
            return result;
        }

        public Volume<T> Tanh()
        {
            var result = BuilderInstance<T>.Volume.SameAs(this.Storage, this.Shape);
            DoTanh(result);
            return result;
        }

        public Volume<T> TanhGradient(Volume<T> input, Volume<T> outputGradient)
        {
            var inputGradient = BuilderInstance<T>.Volume.SameAs(this.Storage, this.Shape);
            DoTanhGradient(input, outputGradient, inputGradient);
            return inputGradient;
        }

        public T[] ToArray()
        {
            return this.Storage.ToArray();
        }

        public override string ToString()
        {
            //Todo: improve
            var sb = new StringBuilder();
            for (var n = 0; n < this.Shape.GetDimension(3); n++)
            {
                for (var c = 0; c < this.Shape.GetDimension(2); c++)
                {
                    sb.Append("[");

                    for (var i = 0; i < this.Shape.GetDimension(1); i++)
                    {
                        sb.Append("[");
                        for (var j = 0; j < this.Shape.GetDimension(0); j++)
                        {
                            sb.Append(Get(j, i, c, n));
                            sb.Append(", ");
                        }

                        sb.Append("],");
                        sb.Append(Environment.NewLine);
                    }

                    sb.Append("],");
                    sb.Append(Environment.NewLine);
                }
            }

            return sb.ToString();
        }
    }
}