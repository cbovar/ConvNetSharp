using System;

namespace ConvNetSharp.Volume.Single
{
    public class Volume : Volume<float>
    {
        internal Volume(float[] array, Shape shape) : this(new NcwhVolumeStorage<float>(array, shape))
        {
        }

        internal Volume(VolumeStorage<float> storage) : base(storage)
        {
        }

        public override void Activation(ActivationType type, Volume<float> volume)
        {
            switch (type)
            {
                case ActivationType.Sigmoid:
                    this.Storage.Map(x => (float)(1.0 / (1.0 + Math.Exp(-x))), volume.Storage);
                    return;
                case ActivationType.Relu:
                    this.Relu(volume);
                    break;
                case ActivationType.Tanh:
                    this.Storage.Map(x => (float)Math.Tanh(x), volume.Storage);
                    return;
                case ActivationType.ClippedRelu:
                    throw new NotImplementedException();
            }
        }

        public override void ActivationGradient(Volume<float> input, Volume<float> outputGradient, ActivationType type, Volume<float> result)
        {
            switch (type)
            {
                case ActivationType.Sigmoid:
                    this.Storage.Map((output, outGradient) => output * (1.0f - output) * outGradient, outputGradient.Storage, result.Storage);
                    return;
                case ActivationType.Relu:
                    this.ReluGradient(input, outputGradient, result);
                    break;
                case ActivationType.Tanh:
                    this.Storage.Map((output, outGradient) => (1.0f - output * output) * outGradient, outputGradient.Storage,
                        result.Storage);
                    break;
                case ActivationType.ClippedRelu:
                    throw new NotImplementedException();
            }
        }

        public override void Add(Volume<float> other, Volume<float> result)
        {
            this.Storage.MapEx((x, y) => x + y, other.Storage, result.Storage);
        }

        public override void Add(Volume<float> result)
        {
            this.Storage.MapEx((x, y) => x + y, result.Storage, result.Storage);
        }

        public override void BiasGradient(Volume<float> result)
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

        public override void Concat(Volume<float> right, Volume<float> result)
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

        public override void Convolution(Volume<float> filters, int pad, int stride, Volume<float> result)
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
                    var y = -pad;
                    for (var ay = 0; ay < outputHeight; y += stride, ay++)
                    {
                        var x = -pad;
                        for (var ax = 0; ax < outputWidth; x += stride, ax++)
                        {
                            // convolve centered at this particular location
                            var a = 0.0f;
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

        public override void ConvolutionGradient(Volume<float> filters, Volume<float> outputGradients,
            Volume<float> filterGradient, int pad,
            int stride,
            Volume<float> inputGradient)
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
                    var y = -pad;
                    for (var ay = 0; ay < outputHeight; y += stride, ay++)
                    {
                        // xyStride
                        var x = -pad;
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
                                            filterGradient.Get(fx, fy, fd, depth) + this.Get(ox, oy, fd, n) * chainGradient);
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

        public override void Divide(Volume<float> other, Volume<float> result)
        {
            this.Storage.MapEx((left, right) => left / right, other.Storage, result.Storage);
        }

        public override void Dropout(float dropProbability, Volume<float> result)
        {
            if (((NcwhVolumeStorage<float>)this.Storage).Dropped == null || ((NcwhVolumeStorage<float>)this.Storage).Dropped.Length != this.Shape.TotalLength)
            {
                ((NcwhVolumeStorage<float>)this.Storage).Dropped = new bool[this.Shape.TotalLength];
            }

            if (dropProbability > 0.0f)
            {
                // do dropout
                this.Storage.Map((x, i) =>
                {
                    var nextDouble = RandomUtilities.NextDouble();
                    if (nextDouble < dropProbability)
                    {
                        ((NcwhVolumeStorage<float>)this.Storage).Dropped[i] = true;
                        return 0;
                    }

                    ((NcwhVolumeStorage<float>)this.Storage).Dropped[i] = false;
                    return x / (1 - dropProbability); // Scale up so that magnitude remains constant accross training and testing
                }, result.Storage);
            }
            else
            {
                this.Storage.Map(x => x, result.Storage);
            }
        }

        public override void DropoutGradient(Volume<float> input, Volume<float> outputGradient, float dropProbability, Volume<float> inputGradient)
        {
            outputGradient.Storage.Map((x, i) =>
            {
                if (((NcwhVolumeStorage<float>)input.Storage).Dropped[i])
                {
                    return 0;
                }

                return x / (1.0f - dropProbability);
            }, inputGradient.Storage);
        }

        public override void Exp(Volume<float> result)
        {
            this.Storage.Map(x => (float)Math.Exp(x), result.Storage);
        }

        public override void Extract(int length, int offset, Volume<float> result)
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

        public override void LeakyRelu(float alpha, Volume<float> volume)
        {
            this.Storage.Map(x => x > 0 ? x : alpha * x, volume.Storage);
        }

        public override void LeakyReluGradient(Volume<float> outputGradient, Volume<float> inputGradient, float alpha)
        {
            this.Storage.Map((x, y) => x >= 0 ? y : y * alpha, outputGradient.Storage, inputGradient.Storage);
        }

        public override void Log(Volume<float> result)
        {
            this.Storage.Map(x => (float)Math.Log(x), result.Storage);
        }

        public override void MatMultiply(Volume<float> right, Volume<float> result)
        {
            if (this.Shape.Dimensions[2] != 1 || right.Shape.Dimensions[2] != 1)
            {
                throw new ArgumentException($"Left and right volumes should be [w, h, 1, b]. left = {this.Shape} right = {right.Shape}");
            }

            var broadCastLeft = this.Shape.Dimensions[3] == 1;
            var broadCastRight = right.Shape.Dimensions[3] == 1;
            if (this.Shape.Dimensions[3] != right.Shape.Dimensions[3] && !(broadCastLeft || broadCastRight))
            {
                throw new ArgumentException($"Left and right volumes should have the same batch size. left = {this.Shape.Dimensions[3]} right = {right.Shape.Dimensions[3]}");
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
                        var cell = 0.0f;
                        for (var k = 0; k < this.Shape.Dimensions[0]; k++)
                        {
                            cell += this.Get(k, j, 0, broadCastLeft ? 0 : n) * right.Get(i, k, 0, broadCastRight ? 0 : n);
                        }

                        result.Set(i, j, 0, n, cell);
                    }
                }
            }
        }

        public override void Max(Volume<float> result)
        {
            var batchSize = this.Shape.Dimensions[3];
            var reshape = this.ReShape(-1, batchSize);

            var n = reshape.Shape.Dimensions[0];

            for (var i = 0; i < batchSize; i++)
            {
                var max = float.MinValue;

                for (var j = 0; j < n; j++)
                {
                    var d = reshape.Get(j, i);
                    if (d > max)
                    {
                        max = d;
                    }
                }

                result.Set(new[] { i }, max);
            }
        }

        public override void Min(Volume<float> result)
        {
            var batchSize = this.Shape.Dimensions[3];
            var reshape = this.ReShape(-1, batchSize);

            var n = reshape.Shape.Dimensions[0];

            for (var i = 0; i < batchSize; i++)
            {
                var min = float.MaxValue;

                for (var j = 0; j < n; j++)
                {
                    var d = reshape.Get(j, i);
                    if (d < min)
                    {
                        min = d;
                    }
                }

                result.Set(new[] { i }, min);
            }
        }

        public override void Multiply(float factor, Volume<float> result)
        {
            this.Storage.Map(x => x * factor, result.Storage);
        }

        public override void Multiply(Volume<float> right, Volume<float> result)
        {
            this.Storage.MapEx((x, y) => x * y, right.Storage, result.Storage);
        }

        public override void Negate(Volume<float> result)
        {
            this.Multiply(-1.0f, result);
        }

        public override void Norm1(Volume<float> result)
        {
            var batchSize = this.Shape.Dimensions[3];
            var reshape = this.ReShape(-1, batchSize);

            var n = reshape.Shape.Dimensions[0];

            for (var i = 0; i < batchSize; i++)
            {
                var sum = 0.0f;

                for (var j = 0; j < n; j++)
                {
                    var d = reshape.Get(j, i);
                    sum += Math.Abs(d);
                }

                result.Set(new[] { i }, sum);
            }
        }

        public override void Pool(int windowWidth, int windowHeight,
            int horizontalPad, int verticalPad, int horizontalStride, int verticalStride, Volume<float> result)
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
                            var a = float.MinValue;

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

        public override void PoolGradient(Volume<float> input, Volume<float> outputGradient,
            int windowWidth, int windowHeight,
            int horizontalPad, int verticalPad, int horizontalStride, int verticalStride,
            Volume<float> inputGradient)
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
                            var a = float.MinValue;
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

        public override void Power(Volume<float> power, Volume<float> result)
        {
            this.Storage.MapEx((x, y) => (float)Math.Pow(x, y), power.Storage, result.Storage);
        }

        public override void Reduce(TensorReduceOp op, Volume<float> result)
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
                    this.Min(result);
                    break;
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

        public override void Relu(Volume<float> volume)
        {
            this.Storage.Map(x => x <= 0 ? 0 : x, volume.Storage);
        }

        public override void ReluGradient(Volume<float> input, Volume<float> output, Volume<float> outputGradient)
        {
            this.Storage.Map((x, y) => x > 0 ? y : 0, output.Storage, outputGradient.Storage);
        }

        public override void Sigmoid(Volume<float> volume)
        {
            this.Storage.Map(x => (float)(1.0 / (1.0 + Math.Exp(-x))), volume.Storage);
        }

        public override void SigmoidGradient(Volume<float> input, Volume<float> outputGradient, Volume<float> inputGradient)
        {
            this.Storage.Map((output, outGradient) => output * (1.0f - output) * outGradient, outputGradient.Storage, inputGradient.Storage);
        }

        public override void Softmax(Volume<float> result)
        {
            var batchSize = this.Shape.Dimensions[3];

            var outputWidth = this.Shape.Dimensions[0];
            var outputHeight = this.Shape.Dimensions[1];
            var outputDepth = this.Shape.Dimensions[2];

            for (var n = 0; n < batchSize; n++)
            {
                // compute max activation
                var amax = double.MinValue;
                for (var depth = 0; depth < outputDepth; depth++)
                {
                    for (var ay = 0; ay < outputHeight; ay++)
                    {
                        for (var ax = 0; ax < outputWidth; ax++)
                        {
                            var v = this.Get(ax, ay, depth, n);
                            if (v > amax)
                            {
                                amax = v;
                            }
                        }
                    }
                }

                // compute exponentials (carefully to not blow up)
                var es = new double[outputDepth * outputHeight * outputWidth];
                var esum = 0.0;

                for (var depth = 0; depth < outputDepth; depth++)
                {
                    for (var ay = 0; ay < outputHeight; ay++)
                    {
                        for (var ax = 0; ax < outputWidth; ax++)
                        {
                            var e = Math.Exp(this.Get(ax, ay, depth, n) - amax);
                            esum += e;
                            es[ax + ay * outputWidth + depth * outputWidth * outputHeight] = e;
                        }
                    }
                }

                // normalize and output to sum to one
                for (var depth = 0; depth < outputDepth; depth++)
                {
                    for (var ay = 0; ay < outputHeight; ay++)
                    {
                        for (var ax = 0; ax < outputWidth; ax++)
                        {
                            es[ax + ay * outputWidth + depth * outputWidth * outputHeight] /= esum;

                            result.Storage.Set(ax, ay, depth, n,
                                (float)es[ax + ay * outputWidth + depth * outputWidth * outputHeight]);
                        }
                    }
                }
            }
        }

        public override void SoftmaxGradient(Volume<float> outputGradient, Volume<float> inputGradient)
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

                    if (yi == 1.0f)
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
                        inputGradientReshape.Set(i, b, pj * (1.0f - pj));
                    }
                    else
                    {
                        inputGradientReshape.Set(i, b, -pj * pi);
                    }
                }
            }
        }

        public override void Sqrt(Volume<float> result)
        {
            this.Storage.Map(x => (float)Math.Sqrt(x), result.Storage);
        }

        public override void SubtractFrom(Volume<float> other, Volume<float> result)
        {
            this.Storage.MapEx((x, y) => y - x, other.Storage, result.Storage);
        }

        public override void Sum(Volume<float> result)
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

        public override void Tanh(Volume<float> volume)
        {
            this.Storage.Map(x => (float)Math.Tanh(x), volume.Storage);
        }

        public override void TanhGradient(Volume<float> input, Volume<float> outputGradient, Volume<float> inputGradient)
        {
            this.Storage.Map((output, outGradient) => (1.0f - output * output) * outGradient, outputGradient.Storage, inputGradient.Storage);
        }

        public override void Tile(Volume<float> reps, Volume<float> result)
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

        public override void Transpose(Volume<float> result)
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
}