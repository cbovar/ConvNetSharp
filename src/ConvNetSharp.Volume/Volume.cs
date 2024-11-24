using System;
using System.Diagnostics;
using System.Numerics;
using System.Text;

namespace ConvNetSharp.Volume;

/// <summary>
///     A Volume (also called tensor in other libraries) is a data container. It has a type (T), a shape and a storage.
/// </summary>
/// <typeparam name="T"></typeparam>
[DebuggerDisplay("Volume {Shape.PrettyPrint()}")]
public abstract class Volume<T> : IDisposable where T : INumber<T>, IMinMaxValue<T>
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

    public virtual void ActivationGradient(Volume<T> input, Volume<T> outputGradient, ActivationType type,
        Volume<T> result)
    {
        switch (type)
        {
            case ActivationType.Sigmoid:
                this.Storage.Map((output, outGradient) => output * (T.One - output) * outGradient,
                    outputGradient.Storage, result.Storage);
                return;
            case ActivationType.Relu:
                this.ReluGradient(input, outputGradient, result);
                break;
            case ActivationType.Tanh:
                this.Storage.Map((output, outGradient) => (T.One - output * output) * outGradient,
                    outputGradient.Storage,
                    result.Storage);
                break;
            case ActivationType.ClippedRelu:
                throw new NotImplementedException();
        }
    }

    public virtual void Add(Volume<T> other, Volume<T> result)
    {
        this.Storage.MapEx((x, y) => x + y, other.Storage, result.Storage);
    }

    /// <summary>
    ///     result = result + this
    /// </summary>
    /// <param name="result"></param>
    public virtual void Add(Volume<T> result)
    {
        this.Storage.MapEx((x, y) => x + y, result.Storage, result.Storage);
    }

    public virtual void BiasGradient(Volume<T> result)
    {
        var batchSize = this.Shape.Dimensions[3];

        var outputWidth = this.Shape.Dimensions[0];
        var outputHeight = this.Shape.Dimensions[1];
        var outputDepth = this.Shape.Dimensions[2];

        for (var n = 0; n < batchSize; n++)
        {
            for (var depth = 0; depth < outputDepth; depth++)
            {
                for (var ay = 0; ay < outputHeight; ay++)
                {
                    for (var ax = 0; ax < outputWidth; ax++)
                    {
                        var chainGradient = this.Get(ax, ay, depth, n);

                        result.Storage.Set(0, 0, depth,
                            result.Storage.Get(0, 0, depth) + chainGradient);
                    }
                }
            }
        }
    }

    public void Clear()
    {
        this.Storage.Clear();
    }

    public Volume<T> Clone()
    {
        var data = new T[this.Shape.TotalLength];
        Array.Copy(this.ToArray(), data, data.Length);

        return BuilderInstance<T>.Volume.From(data, this.Shape);
    }

    public static Shape ComputeConvolutionShape(Shape inputShape, Shape filterShape, int pad, int stride)
    {
        var outputDepth = filterShape.Dimensions[3];
        var outputWidth =
            (int)Math.Floor((inputShape.Dimensions[0] + pad * 2 - filterShape.Dimensions[0]) / (double)stride + 1);
        var outputHeight =
            (int)Math.Floor((inputShape.Dimensions[1] + pad * 2 - filterShape.Dimensions[1]) / (double)stride + 1);

        return new Shape(outputWidth, outputHeight, outputDepth, inputShape.Dimensions[3]);
    }

    public static Shape ComputePoolShape(Shape inputShape, int windowWidth, int windowHeight, int horizontalPad,
        int verticalPad, int horizontalStride, int verticalStride)
    {
        var outputN = inputShape.Dimensions[3];
        var outputDepth = inputShape.Dimensions[2];
        var outputWidth =
            (int)Math.Floor((inputShape.Dimensions[0] + horizontalPad * 2 - windowWidth) / (double)horizontalStride +
                            1);
        var outputHeight =
            (int)Math.Floor((inputShape.Dimensions[1] + verticalPad * 2 - windowHeight) / (double)verticalStride + 1);

        return new Shape(outputWidth, outputHeight, outputDepth, outputN);
    }

    /// <summary>
    ///     Compute expected 2D matrix multiplication result shape
    ///     [K, M, 1, BatchSize] x [N, K, 1, BatchSize] => [N, M, 1, BatchSize]
    /// </summary>
    /// <param name="leftShape">left 2D matrix / volume</param>
    /// <param name="rightShape">right 2D matrix / volume</param>
    /// <returns></returns>
    public static Shape ComputeMatMultiplyShape(Shape leftShape, Shape rightShape)
    {
        var batchSize = Math.Max(leftShape.Dimensions[3], rightShape.Dimensions[3]);

        return new Shape(rightShape.Dimensions[0], leftShape.Dimensions[1], 1, batchSize);
    }

    public virtual void Concat(Volume<T> right, Volume<T> result)
    {
        var batchSize = Math.Max(this.Shape.Dimensions[3], right.Shape.Dimensions[3]);

        if (this.Shape.TotalLength > 1 && right.Shape.TotalLength > 1)
        {
            var left = this.ReShape(new Shape(1, 1, -1, batchSize));
            right = right.ReShape(new Shape(1, 1, -1, batchSize));

            var elementPerBatch = result.Shape.TotalLength / batchSize;
            var threshold = left.Shape.Dimensions[2];

            for (var n = 0; n < batchSize; n++)
            {
                for (var i = 0; i < elementPerBatch; i++)
                {
                    result.Set(0, 0, i, n, i < threshold ? left.Get(0, 0, i, n) : right.Get(0, 0, i - threshold, n));
                }
            }
        }
        else if (this.Shape.TotalLength == 1 && right.Shape.TotalLength > 1)
        {
            // Left volume is actually a scalar => broadcast its value

            right = right.ReShape(new Shape(1, 1, -1, batchSize));
            var elementPerBatch = result.Shape.TotalLength / batchSize;
            const int threshold = 1;

            for (var n = 0; n < batchSize; n++)
            {
                for (var i = 0; i < elementPerBatch; i++)
                {
                    result.Set(0, 0, i, n, i < threshold ? this.Get(0) : right.Get(0, 0, i - threshold, n));
                }
            }
        }
        else
        {
            // Right volume is actually a scalar => broadcast its value

            var left = this.ReShape(new Shape(1, 1, -1, batchSize));
            var elementPerBatch = result.Shape.TotalLength / batchSize;
            var threshold = left.Shape.Dimensions[2];

            for (var n = 0; n < batchSize; n++)
            {
                for (var i = 0; i < elementPerBatch; i++)
                {
                    result.Set(0, 0, i, n, i < threshold ? left.Get(0, 0, i, n) : right.Get(0));
                }
            }
        }
    }

    public virtual void Convolution(Volume<T> filters, int pad, int stride, Volume<T> result)
    {
        this.Convolution(filters, pad, pad, stride, result);
    }

    public virtual void Convolution(Volume<T> filters, int xpad, int ypad, int stride, Volume<T> result)
    {
        var batchSize = this.Shape.Dimensions[3];

        var inputWidth = this.Shape.Dimensions[0];
        var inputHeight = this.Shape.Dimensions[1];

        var outputWidth = result.Shape.Dimensions[0];
        var outputHeight = result.Shape.Dimensions[1];
        var outputDepth = result.Shape.Dimensions[2];

        var filterWidth = filters.Shape.Dimensions[0];
        var filterHeight = filters.Shape.Dimensions[1];
        var filterDepth = filters.Shape.Dimensions[2];

        for (var n = 0; n < batchSize; n++)
        {
            for (var depth = 0; depth < outputDepth; depth++)
            {
                var y = -ypad;
                for (var ay = 0; ay < outputHeight; y += stride, ay++)
                {
                    var x = -xpad;
                    for (var ax = 0; ax < outputWidth; x += stride, ax++)
                    {
                        // convolve centered at this particular location
                        var a = T.Zero;
                        for (var fy = 0; fy < filterHeight; fy++)
                        {
                            var oy = y + fy; // coordinates in the original input array coordinates
                            for (var fx = 0; fx < filterWidth; fx++)
                            {
                                var ox = x + fx;
                                if (oy >= 0 && oy < inputHeight && ox >= 0 && ox < inputWidth)
                                {
                                    for (var fd = 0; fd < filterDepth; fd++)
                                    {
                                        a += filters.Storage.Get(fx, fy, fd, depth) *
                                             this.Storage.Get(ox, oy, fd, n);
                                    }
                                }
                            }
                        }

                        result.Storage.Set(ax, ay, depth, n, a);
                    }
                }
            }
        }
    }

    public virtual void ConvolutionGradient(Volume<T> filters, Volume<T> outputGradients,
        Volume<T> filterGradient, int pad, int stride, Volume<T> inputGradient)
    {
        this.ConvolutionGradient(filters, outputGradients, filterGradient, pad, pad, stride, inputGradient);
    }

    public virtual void ConvolutionGradient(Volume<T> filters, Volume<T> outputGradients,
        Volume<T> filterGradient, int xpad, int ypad, int stride, Volume<T> inputGradient)
    {
        inputGradient.Clear(); // zero out gradient wrt bottom data, we're about to fill it

        var batchSize = this.Shape.Dimensions[3];

        var inputWidth = this.Shape.Dimensions[0];
        var inputHeight = this.Shape.Dimensions[1];

        var outputWidth = outputGradients.Shape.Dimensions[0];
        var outputHeight = outputGradients.Shape.Dimensions[1];
        var outputDepth = outputGradients.Shape.Dimensions[2];

        var filterWidth = filters.Shape.Dimensions[0];
        var filterHeight = filters.Shape.Dimensions[1];
        var filterDepth = filters.Shape.Dimensions[2];

        for (var n = 0; n < batchSize; n++)
        {
            for (var depth = 0; depth < outputDepth; depth++)
            {
                var y = -ypad;
                for (var ay = 0; ay < outputHeight; y += stride, ay++)
                {
                    // xyStride
                    var x = -xpad;
                    for (var ax = 0; ax < outputWidth; x += stride, ax++)
                    {
                        // xyStride

                        // convolve centered at this particular location
                        var chainGradient = outputGradients.Get(ax, ay, depth, n);

                        // gradient from above, from chain rule
                        for (var fy = 0; fy < filterHeight; fy++)
                        {
                            var oy = y + fy; // coordinates in the original input array coordinates
                            for (var fx = 0; fx < filterWidth; fx++)
                            {
                                var ox = x + fx;
                                if (oy < 0 || oy >= inputHeight || ox < 0 || ox >= inputWidth)
                                {
                                    continue;
                                }

                                for (var fd = 0; fd < filterDepth; fd++)
                                {
                                    filterGradient.Storage.Set(fx, fy, fd, depth,
                                        filterGradient.Get(fx, fy, fd, depth) +
                                        this.Get(ox, oy, fd, n) * chainGradient);
                                    inputGradient.Storage.Set(ox, oy, fd, n,
                                        inputGradient.Storage.Get(ox, oy, fd, n) +
                                        filters.Get(fx, fy, fd, depth) * chainGradient);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    public virtual void Divide(Volume<T> other, Volume<T> result)
    {
        this.Storage.MapEx((left, right) => left / right, other.Storage, result.Storage);
    }

    /// <summary>
    ///     Computes dropout. Result will be scaled up by 1 / (1 - dropProbability).
    /// </summary>
    /// <param name="dropProbability">Probability at which elements will be set to 0</param>
    /// <param name="result">Output volume</param>
    public virtual void Dropout(float dropProbability, Volume<T> result)
    {
        if (((NcwhVolumeStorage<T>)this.Storage).Dropped == null ||
            ((NcwhVolumeStorage<T>)this.Storage).Dropped.Length != this.Shape.TotalLength)
        {
            ((NcwhVolumeStorage<T>)this.Storage).Dropped = new bool[this.Shape.TotalLength];
        }

        if (dropProbability > 0.0f)
        {
            // do dropout
            this.Storage.Map((x, i) =>
            {
                var nextDouble = RandomUtilities.NextDouble();
                if (nextDouble < dropProbability)
                {
                    ((NcwhVolumeStorage<T>)this.Storage).Dropped[i] = true;
                    return T.Zero;
                }

                ((NcwhVolumeStorage<T>)this.Storage).Dropped[i] = false;
                return
                    x / (T.One -
                         T.CreateChecked(
                             dropProbability)); // Scale up so that magnitude remains constant across training and testing
            }, result.Storage);
        }
        else
        {
            this.Storage.Map(x => x, result.Storage);
        }
    }

    public virtual void DropoutGradient(Volume<T> input, Volume<T> outputGradient, float dropProbability,
        Volume<T> inputGradient)
    {
        outputGradient.Storage.Map((x, i) =>
        {
            if (((NcwhVolumeStorage<T>)input.Storage).Dropped[i])
            {
                return T.Zero;
            }

            return x / (T.One - T.CreateChecked(dropProbability));
        }, inputGradient.Storage);
    }

    public abstract void Exp(Volume<T> result);

    public virtual void Extract(int length, int offset, Volume<T> result)
    {
        var input = this.ReShape(1, 1, Shape.None, Shape.Keep);

        if (input.Shape.TotalLength == 1)
        {
            var v = input.Get(0);
            this.Storage.Map(x => v, result.Storage);
        }
        else
        {
            var batchSize = this.Shape.Dimensions[3];
            for (var n = 0; n < batchSize; n++)
            {
                for (var i = 0; i < length; i++)
                {
                    result.Set(0, 0, i, n, input.Get(0, 0, i + offset, n));
                }
            }
        }
    }

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

    public virtual void LeakyRelu(T alpha, Volume<T> result)
    {
        this.Storage.Map(x => x > T.Zero ? x : alpha * x, result.Storage);
    }

    public virtual void LeakyReluGradient(Volume<T> outputGradient, Volume<T> inputGradient, T alpha)
    {
        this.Storage.Map((x, y) => x >= T.Zero ? y : y * alpha, outputGradient.Storage, inputGradient.Storage);
    }

    public abstract void Log(Volume<T> result);

    public void MapInPlace(Func<T, T> f)
    {
        this.Storage.MapInPlace(f);
    }

    public void MapInPlace(Func<T, T, T> f, Volume<T> other)
    {
        this.Storage.MapInPlace(f, other.Storage);
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
    public virtual void MatMultiply(Volume<T> right, Volume<T> result)
    {
        if (this.Shape.Dimensions[2] != 1 || right.Shape.Dimensions[2] != 1)
        {
            throw new ArgumentException(
                $"Left and right volumes should be [w, h, 1, b]. left = {this.Shape} right = {right.Shape}");
        }

        var broadCastLeft = this.Shape.Dimensions[3] == 1;
        var broadCastRight = right.Shape.Dimensions[3] == 1;
        if (this.Shape.Dimensions[3] != right.Shape.Dimensions[3] && !(broadCastLeft || broadCastRight))
        {
            throw new ArgumentException(
                $"Left and right volumes should have the same batch size. left = {this.Shape.Dimensions[3]} right = {right.Shape.Dimensions[3]}");
        }

        var expectedShape = ComputeMatMultiplyShape(this.Shape, right.Shape);

        if (!result.Shape.Equals(expectedShape))
        {
            throw new ArgumentException($"Result shape should be {expectedShape} but is {result.Shape}");
        }

        for (var n = 0; n < this.Shape.Dimensions[3]; n++)
        {
            for (var i = 0; i < expectedShape.Dimensions[0]; i++)
            {
                for (var j = 0; j < expectedShape.Dimensions[1]; j++)
                {
                    var cell = T.Zero;
                    for (var k = 0; k < this.Shape.Dimensions[0]; k++)
                    {
                        cell += this.Get(k, j, 0, broadCastLeft ? 0 : n) * right.Get(i, k, 0, broadCastRight ? 0 : n);
                    }

                    result.Set(i, j, 0, n, cell);
                }
            }
        }
    }

    public virtual void Max(Volume<T> result)
    {
        var batchSize = this.Shape.Dimensions[3];
        var reshape = this.ReShape(-1, batchSize);

        var n = reshape.Shape.Dimensions[0];

        for (var i = 0; i < batchSize; i++)
        {
            var max = T.MinValue;

            for (var j = 0; j < n; j++)
            {
                var d = reshape.Get(j, i);
                if (d > max)
                {
                    max = d;
                }
            }

            result.Set([i], max);
        }
    }

    public virtual void Min(Volume<T> result)
    {
        var batchSize = this.Shape.Dimensions[3];
        var reshape = this.ReShape(-1, batchSize);

        var n = reshape.Shape.Dimensions[0];

        for (var i = 0; i < batchSize; i++)
        {
            var min = T.MaxValue;

            for (var j = 0; j < n; j++)
            {
                var d = reshape.Get(j, i);
                if (d < min)
                {
                    min = d;
                }
            }

            result.Set([i], min);
        }
    }

    public virtual void Multiply(T factor, Volume<T> result)
    {
        this.Storage.Map(x => x * factor, result.Storage);
    }

    public virtual void Multiply(Volume<T> right, Volume<T> result)
    {
        this.Storage.MapEx((x, y) => x * y, right.Storage, result.Storage);
    }

    public virtual void Negate(Volume<T> result)
    {
        this.Multiply(-T.One, result);
    }

    public abstract void Norm1(Volume<T> result);

    public static implicit operator Volume<T>(T t)
    {
        return BuilderInstance<T>.Volume.From([t], new Shape(1));
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

    public virtual void Pool(int windowWidth, int windowHeight,
        int horizontalPad, int verticalPad, int horizontalStride, int verticalStride, Volume<T> result)
    {
        var inputWidth = this.Shape.Dimensions[0];
        var inputHeight = this.Shape.Dimensions[1];

        var outputWidth = result.Shape.Dimensions[0];
        var outputHeight = result.Shape.Dimensions[1];
        var outputDepth = result.Shape.Dimensions[2];
        var batchSize = result.Shape.Dimensions[3];

        for (var n = 0; n < batchSize; n++)
        {
            for (var depth = 0; depth < outputDepth; depth++)
            {
                var x = -horizontalPad;
                for (var ax = 0; ax < outputWidth; x += verticalStride, ax++)
                {
                    var y = -verticalPad;
                    for (var ay = 0; ay < outputHeight; y += horizontalStride, ay++)
                    {
                        var a = T.MinValue;

                        for (var fx = 0; fx < windowWidth; fx++)
                        {
                            for (var fy = 0; fy < windowHeight; fy++)
                            {
                                var oy = y + fy;
                                var ox = x + fx;
                                if (oy < 0 || oy >= inputHeight || ox < 0 || ox >= inputWidth)
                                {
                                    continue;
                                }

                                var v = this.Get(ox, oy, depth, n);
                                // perform max pooling and store pointers to where
                                // the max came from. This will speed up backprop 
                                // and can help make nice visualizations in future
                                if (v > a)
                                {
                                    a = v;
                                }
                            }
                        }

                        result.Storage.Set(ax, ay, depth, n, a);
                    }
                }
            }
        }
    }

    public virtual void PoolGradient(Volume<T> input, Volume<T> outputGradient,
        int windowWidth, int windowHeight,
        int horizontalPad, int verticalPad, int horizontalStride, int verticalStride,
        Volume<T> inputGradient)
    {
        var inputWidth = input.Shape.Dimensions[0];
        var inputHeight = input.Shape.Dimensions[1];

        var outputWidth = outputGradient.Shape.Dimensions[0];
        var outputHeight = outputGradient.Shape.Dimensions[1];
        var outputDepth = outputGradient.Shape.Dimensions[2];
        var batchSize = outputGradient.Shape.Dimensions[3];

        for (var n = 0; n < batchSize; n++)
        {
            for (var depth = 0; depth < outputDepth; depth++)
            {
                var x = -horizontalPad;
                for (var ax = 0; ax < outputWidth; x += verticalStride, ax++)
                {
                    var y = -verticalPad;
                    for (var ay = 0; ay < outputHeight; y += horizontalStride, ay++)
                    {
                        var a = T.MinValue;
                        int winx = -1, winy = -1;

                        for (var fx = 0; fx < windowWidth; fx++)
                        {
                            for (var fy = 0; fy < windowHeight; fy++)
                            {
                                var oy = y + fy;
                                var ox = x + fx;
                                if (oy < 0 || oy >= inputHeight || ox < 0 || ox >= inputWidth)
                                {
                                    continue;
                                }

                                var v = input.Get(ox, oy, depth, n);
                                // perform max pooling and store pointers to where
                                // the max came from. This will speed up backprop 
                                // and can help make nice visualizations in future
                                if (v > a)
                                {
                                    a = v;
                                    winx = ox;
                                    winy = oy;
                                }
                            }
                        }

                        var chainGradient = outputGradient.Get(ax, ay, depth, n);
                        inputGradient.Storage.Set(winx, winy, depth, n, chainGradient);
                    }
                }
            }
        }
    }

    /// <summary>
    ///     result = this ^ power
    /// </summary>
    /// <param name="power">power. It will be broadcasted if scalar</param>
    /// <param name="result">result</param>
    public abstract void Power(Volume<T> power, Volume<T> result);

    public virtual void Reduce(TensorReduceOp op, Volume<T> result)
    {
        if (this.Shape.Equals(result.Shape))
        {
            result.Storage.CopyFrom(this.Storage);
            return;
        }

        switch (op)
        {
            case TensorReduceOp.Add:
                this.Sum(result);
                break;
            case TensorReduceOp.Mul:
                throw new NotImplementedException();
            case TensorReduceOp.Min:
                throw new NotImplementedException();
            case TensorReduceOp.Max:
                this.Max(result);
                break;
            case TensorReduceOp.AMax:
                throw new NotImplementedException();
            case TensorReduceOp.Avg:
                throw new NotImplementedException();
            case TensorReduceOp.Norm1:
                this.Norm1(result);
                break;
            case TensorReduceOp.Norm2:
                throw new NotImplementedException();
            default:
                throw new ArgumentOutOfRangeException(nameof(op), op, null);
        }
    }

    public virtual void Relu(Volume<T> volume)
    {
        this.Storage.Map(x => x <= T.Zero ? T.Zero : x, volume.Storage);
    }

    public virtual void
        ReluGradient(Volume<T> input, Volume<T> outputGradient, Volume<T> inputGradient) // TODO: Check variable names
    {
        this.Storage.Map((x, y) => x > T.Zero ? y : T.Zero, outputGradient.Storage, inputGradient.Storage);
    }

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

        guessedShape.GuessUnknownDimension(this.Shape.TotalLength);

        Count--;

        return BuilderInstance<T>.Volume.Build(this.Storage, guessedShape);
    }

    public Volume<T> ReShape(params int[] dimensions)
    {
        return this.ReShape(Shape.From(dimensions));
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

    public virtual void SigmoidGradient(Volume<T> input, Volume<T> outputGradient, Volume<T> inputGradient)
    {
        this.Storage.Map((output, outGradient) => output * (T.One - output) * outGradient, outputGradient.Storage,
            inputGradient.Storage);
    }

    public abstract void Softmax(Volume<T> result);

    public virtual void SoftmaxGradient(Volume<T> outputGradient, Volume<T> inputGradient)
    {
        var batchSize = this.Shape.Dimensions[3];

        var outputReshape = this.ReShape(-1, batchSize);
        var outputGradientReshape = outputGradient.ReShape(-1, batchSize);
        var inputGradientReshape = inputGradient.ReShape(-1, batchSize);

        var firstDim = outputReshape.Shape.Dimensions[0];

        for (var b = 0; b < batchSize; b++)
        {
            var classIndex = -1;

            for (var i = 0; i < firstDim; i++)
            {
                var yi = outputGradientReshape.Get(i, b);

                if (yi == T.One)
                {
                    classIndex = i;
                }
            }

            var pj = outputReshape.Get(classIndex, b);

            // input gradient:
            // pi(1 - pi) if i = class index
            // -pipj if i != class index
            for (var i = 0; i < firstDim; i++)
            {
                var pi = outputReshape.Get(i, b);

                if (i == classIndex)
                {
                    inputGradientReshape.Set(i, b, pj * (T.One - pj));
                }
                else
                {
                    inputGradientReshape.Set(i, b, -pj * pi);
                }
            }
        }
    }

    public abstract void Sqrt(Volume<T> result);

    public virtual void SubtractFrom(Volume<T> other, Volume<T> result)
    {
        this.Storage.MapEx((x, y) => y - x, other.Storage, result.Storage);
    }

    public virtual void Sum(Volume<T> result)
    {
        var batchSize = this.Shape.Dimensions[3];
        var channel = this.Shape.Dimensions[2];
        var height = this.Shape.Dimensions[1];
        var width = this.Shape.Dimensions[0];

        var resultWIsOne = result.Shape.Dimensions[0] == 1;
        var resultHIsOne = result.Shape.Dimensions[1] == 1;
        var resultCIsOne = result.Shape.Dimensions[2] == 1;
        var resultNIsOne = result.Shape.Dimensions[3] == 1;

        for (var n = 0; n < batchSize; n++)
        {
            for (var c = 0; c < channel; c++)
            {
                for (var h = 0; h < height; h++)
                {
                    for (var w = 0; w < width; w++)
                    {
                        var val = this.Get(w, h, c, n);

                        var resultW = resultWIsOne ? 0 : w;
                        var resultH = resultHIsOne ? 0 : h;
                        var resultC = resultCIsOne ? 0 : c;
                        var resultN = resultNIsOne ? 0 : n;

                        var current = result.Get(resultW, resultH, resultC, resultN);
                        result.Set(resultW, resultH, resultC, resultN, current + val);
                    }
                }
            }
        }
    }

    public abstract void Tanh(Volume<T> result);

    public virtual void TanhGradient(Volume<T> input, Volume<T> outputGradient, Volume<T> inputGradient)
    {
        this.Storage.Map((output, outGradient) => (T.One - output * output) * outGradient, outputGradient.Storage,
            inputGradient.Storage);
    }

    public virtual void Tile(Volume<T> reps, Volume<T> result)
    {
        var batchSize = this.Shape.Dimensions[3];
        var channel = this.Shape.Dimensions[2];
        var height = this.Shape.Dimensions[1];
        var width = this.Shape.Dimensions[0];

        var outputBatchSize = result.Shape.Dimensions[3];
        var outputChannel = result.Shape.Dimensions[2];
        var outputHeight = result.Shape.Dimensions[1];
        var outputWidth = result.Shape.Dimensions[0];

        for (var n = 0; n < outputBatchSize; n++)
        {
            for (var c = 0; c < outputChannel; c++)
            {
                for (var h = 0; h < outputHeight; h++)
                {
                    for (var w = 0; w < outputWidth; w++)
                    {
                        result.Set(w, h, c, n, this.Get(w % width, h % height, c % channel, n % batchSize));
                    }
                }
            }
        }
    }

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
                        sb.Append(this.Get(j, i, c, n));
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
    ///     Flips a 2D volume over its diagonal
    ///     [i, j, 0, batch] => [j, i, 0, batch]
    /// </summary>
    /// <param name="result"></param>
    public virtual void Transpose(Volume<T> result)
    {
        var expectedShape = new Shape(this.Shape.Dimensions[1], this.Shape.Dimensions[0], 1, this.Shape.Dimensions[3]);

        if (!result.Shape.Equals(expectedShape))
        {
            throw new ArgumentException($"Result shape should be {expectedShape}");
        }

        for (var n = 0; n < this.Shape.Dimensions[3]; n++)
        {
            for (var j = 0; j < this.Shape.Dimensions[1]; j++)
            {
                for (var i = 0; i < this.Shape.Dimensions[0]; i++)
                {
                    result.Set(j, i, 0, n, this.Get(i, j, 0, n));
                }
            }
        }
    }
}