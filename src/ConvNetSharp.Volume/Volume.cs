using System;
using System.Diagnostics;
using System.Text;

namespace ConvNetSharp.Volume
{
    /// <summary>
    ///     A Volume (also called tensor in other librairies) is a data container. It has a type (T), a shape and a storage.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    [DebuggerDisplay("Volume {Shape.PrettyPrint()}")]
    public abstract class Volume<T> : IDisposable
        where T : struct, IEquatable<T>, IFormattable
    {
        public static int Count;

        protected Volume(VolumeStorage<T> storage)
        {
            Count++;

            this.Storage = storage;
        }

        public VolumeStorage<T> Storage { get; }

        public Shape Shape => this.Storage.Shape;

        public virtual void Dispose()
        {
            if (this.Storage is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }

        public abstract void Activation(ActivationType type, Volume<T> result);

        public abstract void ActivationGradient(Volume<T> input, Volume<T> outputGradient, ActivationType type, Volume<T> result);

        public abstract void Add(Volume<T> other, Volume<T> result);

        /// <summary>
        ///     result = result + this
        /// </summary>
        /// <param name="result"></param>
        public abstract void Add(Volume<T> result);

        public abstract void BiasGradient(Volume<T> result);

        public void Clear()
        {
            this.Storage.Clear();
        }

        public Volume<T> Clone()
        {
            var data = new T[this.Shape.TotalLength];
            Array.Copy(ToArray(), data, data.Length);

            return BuilderInstance<T>.Volume.From(data, this.Shape);
        }

        public static Shape ComputeConvolutionShape(Shape inputShape, Shape filterShape, int pad, int stride)
        {
            var outputDepth = filterShape.Dimensions[3];
            var outputWidth = (int)Math.Floor((inputShape.Dimensions[0] + pad * 2 - filterShape.Dimensions[0]) / (double)stride + 1);
            var outputHeight = (int)Math.Floor((inputShape.Dimensions[1] + pad * 2 - filterShape.Dimensions[1]) / (double)stride + 1);

            return new Shape(outputWidth, outputHeight, outputDepth, inputShape.Dimensions[3]);
        }

        public static Shape ComputePoolShape(Shape inputShape, int windowWidth, int windowHeight, int horizontalPad, int verticalPad, int horizontalStride, int verticalStride)
        {
            var outputN = inputShape.Dimensions[3];
            var outputDepth = inputShape.Dimensions[2];
            var outputWidth = (int)Math.Floor((inputShape.Dimensions[0] + horizontalPad * 2 - windowWidth) / (double)horizontalStride + 1);
            var outputHeight = (int)Math.Floor((inputShape.Dimensions[1] + verticalPad * 2 - windowHeight) / (double)verticalStride + 1);

            return new Shape(outputWidth, outputHeight, outputDepth, outputN);
        }

        /// <summary>
        /// Compute expected 2D matrix multiplication result shape 
        /// [K, M, 1, BatchSize] x [N, K, 1, BatchSize] => [N, M, 1, BatchSize]
        /// </summary>
        /// <param name="leftShape">left 2D matrix / volume</param>
        /// <param name="rightShape">right 2D matrix / volume</param>
        /// <returns></returns>
        public static Shape ComputeMatMultiplyShape(Shape leftShape, Shape rightShape)
        {
            var batchSize = Math.Max(leftShape.Dimensions[3], rightShape.Dimensions[3]);

            return new Shape(rightShape.Dimensions[0], leftShape.Dimensions[1], 1, batchSize);
        }

        public abstract void Concat(Volume<T> right, Volume<T> result);

        public abstract void Convolution(Volume<T> filters, int pad, int stride, Volume<T> result);

        public abstract void ConvolutionGradient(Volume<T> filters, Volume<T> outputGradients,
            Volume<T> filterGradient, int pad, int stride, Volume<T> inputGradient);

        public abstract void Divide(Volume<T> other, Volume<T> result);

        /// <summary>
        ///     Computes dropout. Result will be scaled up by 1 / (1 - dropProbability).
        /// </summary>
        /// <param name="dropProbability">Probability at which elements will be set to 0</param>
        /// <param name="result">Output volume</param>
        public abstract void Dropout(T dropProbability, Volume<T> result);

        public abstract void DropoutGradient(Volume<T> input, Volume<T> outputGradient, T dropProbability, Volume<T> inputGradient);

        public abstract void Exp(Volume<T> result);

        public abstract void Extract(int length, int offset, Volume<T> result);

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

        public abstract void LeakyRelu(T alpha, Volume<T> result);

        public abstract void LeakyReluGradient(Volume<T> outputGradient, Volume<T> inputGradient, T alpha);

        public abstract void Log(Volume<T> result);

        public void MapInplace(Func<T, T> f)
        {
            this.Storage.MapInplace(f);
        }

        public void MapInplace(Func<T, T, T> f, Volume<T> other)
        {
            this.Storage.MapInplace(f, other.Storage);
        }

        /// <summary>
        ///     Matrix multiplication
        ///     left (this) x right = result
        ///     Where left is a 2D volume of shape [K, M, 1, batchsize]
        ///     right is a 2D volume of shape [N, K, 1, batchsize]
        ///     and result is a 2D volume of shape [N, M, 1, batchsize]
        /// </summary>
        /// <param name="right">2D volume of shape [N, K, 1, batchsize]</param>
        /// <param name="result">2D volume of shape [N, M, 1, batchsize]</param>
        public abstract void MatMultiply(Volume<T> right, Volume<T> result);

        public abstract void Max(Volume<T> result);

        public abstract void Min(Volume<T> result);

        public abstract void Multiply(T factor, Volume<T> result);

        public abstract void Multiply(Volume<T> right, Volume<T> result);

        public abstract void Negate(Volume<T> result);

        public abstract void Norm1(Volume<T> result);

        public static implicit operator Volume<T>(T t)
        {
            return BuilderInstance<T>.Volume.From(new[] { t }, new Shape(1));
        }

        public static implicit operator Volume<T>(T[] t)
        {
            return BuilderInstance<T>.Volume.From(t, new Shape(1, 1, t.Length, 1));
        }

        public static implicit operator T(Volume<T> v)
        {
            if (v.Shape.TotalLength == 1)
            {
                return v.Get(0);
            }

            throw new ArgumentException($"Volume should have a Shape [1] to be converter to a {typeof(T)}", nameof(v));
        }

        public abstract void Pool(int windowWidth, int windowHeight,
            int horizontalPad, int verticalPad, int horizontalStride, int verticalStride, Volume<T> result);

        public abstract void PoolGradient(Volume<T> input, Volume<T> outputGradient,
            int windowWidth, int windowHeight,
            int horizontalPad, int verticalPad, int horizontalStride, int verticalStride,
            Volume<T> inputGradient);

        /// <summary>
        ///     result = this ^ power
        /// </summary>
        /// <param name="power">power. It will be broadcasted if scalar</param>
        /// <param name="result">result</param>
        public abstract void Power(Volume<T> power, Volume<T> result);

        public abstract void Reduce(TensorReduceOp op, Volume<T> result);

        public abstract void Relu(Volume<T> result);

        public abstract void ReluGradient(Volume<T> input, Volume<T> outputGradient, Volume<T> inputGradient);

        public Volume<T> ReShape(Shape shape)
        {
            var guessedShape = new Shape(shape);
            for (var i = 0; i < 4; i++)
            {
                if (shape.Dimensions[i] == Shape.Keep)
                {
                    guessedShape.SetDimension(i, this.Shape.Dimensions[i]);
                }
            }

            guessedShape.GuessUnkownDimension(this.Shape.TotalLength);

            Count--;

            return BuilderInstance<T>.Volume.Build(this.Storage, guessedShape);
        }

        public Volume<T> ReShape(params int[] dimensions)
        {
            return ReShape(Shape.From(dimensions));
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

        public abstract void Sigmoid(Volume<T> result);

        public abstract void SigmoidGradient(Volume<T> input, Volume<T> outputGradient, Volume<T> inputGradient);

        public abstract void Softmax(Volume<T> result);

        public abstract void SoftmaxGradient(Volume<T> outputGradient, Volume<T> inputGradient);

        public abstract void Sqrt(Volume<T> result);

        public abstract void SubtractFrom(Volume<T> other, Volume<T> result);

        public abstract void Sum(Volume<T> result);

        public abstract void Tanh(Volume<T> result);

        public abstract void TanhGradient(Volume<T> input, Volume<T> outputGradient, Volume<T> inputGradient);

        public abstract void Tile(Volume<T> reps, Volume<T> result);

        public T[] ToArray()
        {
            return this.Storage.ToArray();
        }

        public override string ToString()
        {
            //Todo: improve
            var sb = new StringBuilder();
            for (var n = 0; n < this.Shape.Dimensions[3]; n++)
            {
                for (var c = 0; c < this.Shape.Dimensions[2]; c++)
                {
                    sb.Append("[");

                    for (var i = 0; i < this.Shape.Dimensions[1]; i++)
                    {
                        sb.Append("[");
                        for (var j = 0; j < this.Shape.Dimensions[0]; j++)
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

        /// <summary>
        /// Flips a 2D volume over its diagonal
        /// [i, j, 0, batch] => [j, i, 0, batch]
        /// </summary>
        /// <param name="result"></param>
        public abstract void Transpose(Volume<T> result);
    }
}