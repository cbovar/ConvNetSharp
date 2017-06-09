using System;

namespace ConvNetSharp.Volume.Single
{
    public class Volume : Volume<float>
    {
        public Volume(float[] array, Shape shape) : this(new NcwhVolumeStorage<float>(array, shape))
        {
        }

        public Volume(VolumeStorage<float> storage) : base(storage)
        {
        }

        public override void DoAdd(Volume<float> other, Volume<float> result)
        {
            this.Storage.MapEx((x, y) => x + y, other.Storage, result.Storage);
        }

        protected override void DoBiasGradient(Volume<float> biasGradient)
        {
            var batchSize = this.Shape.GetDimension(3);

            var outputWidth = this.Shape.GetDimension(0);
            var outputHeight = this.Shape.GetDimension(1);
            var outputDepth = this.Shape.GetDimension(2);

            for (var n = 0; n < batchSize; n++)
            {
                for (var depth = 0; depth < outputDepth; depth++)
                {
                    for (var ay = 0; ay < outputHeight; ay++)
                    {
                        for (var ax = 0; ax < outputWidth; ax++)
                        {
                            var chainGradient = Get(ax, ay, depth, n);

                            biasGradient.Storage.Set(0, 0, depth,
                                biasGradient.Storage.Get(0, 0, depth) + chainGradient);
                        }
                    }
                }
            }
        }

        public override void DoConvolution(Volume<float> filters, int pad, int stride, Volume<float> result)
        {
            var batchSize = this.Shape.GetDimension(3);

            var inputWidth = this.Shape.GetDimension(0);
            var inputHeight = this.Shape.GetDimension(1);

            var outputWidth = result.Shape.GetDimension(0);
            var outputHeight = result.Shape.GetDimension(1);
            var outputDepth = result.Shape.GetDimension(2);

            var filterWidth = filters.Shape.GetDimension(0);
            var filterHeight = filters.Shape.GetDimension(1);
            var filterDepth = filters.Shape.GetDimension(2);

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

        protected override void DoConvolutionGradient(Volume<float> filters, Volume<float> outputGradients,
            Volume<float> inputGradient, Volume<float> filterGradient, int pad,
            int stride)
        {
            inputGradient.Clear(); // zero out gradient wrt bottom data, we're about to fill it

            var batchSize = this.Shape.GetDimension(3);

            var inputWidth = this.Shape.GetDimension(0);
            var inputHeight = this.Shape.GetDimension(1);

            var outputWidth = outputGradients.Shape.GetDimension(0);
            var outputHeight = outputGradients.Shape.GetDimension(1);
            var outputDepth = outputGradients.Shape.GetDimension(2);

            var filterWidth = filters.Shape.GetDimension(0);
            var filterHeight = filters.Shape.GetDimension(1);
            var filterDepth = filters.Shape.GetDimension(2);

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
                                    if (oy >= 0 && oy < inputHeight && ox >= 0 && ox < inputWidth)
                                    {
                                        for (var fd = 0; fd < filterDepth; fd++)
                                        {
                                            filterGradient.Storage.Set(fx, fy, fd, depth,
                                                filterGradient.Get(fx, fy, fd, depth) +
                                                Get(ox, oy, fd, n) * chainGradient);
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
        }

        public override void DoMultiply(Volume<float> result, float factor)
        {
            this.Storage.Map(x => x * factor, result.Storage);
        }

        protected override void DoNegate(Volume<float> volume)
        {
            DoMultiply(volume, -1.0f);
        }

        public override void DoPool(Volume<float> result, int windowWidth, int windowHeight,
            int horizontalPad, int verticalPad, int horizontalStride, int verticalStride)
        {
            var inputWidth = this.Shape.GetDimension(0);
            var inputHeight = this.Shape.GetDimension(1);

            var outputWidth = result.Shape.GetDimension(0);
            var outputHeight = result.Shape.GetDimension(1);
            var outputDepth = result.Shape.GetDimension(2);
            var batchSize = result.Shape.GetDimension(3);

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
                                    if (oy >= 0 && oy < inputHeight && ox >= 0 && ox < inputWidth)
                                    {
                                        var v = Get(ox, oy, depth, n);
                                        // perform max pooling and store pointers to where
                                        // the max came from. This will speed up backprop 
                                        // and can help make nice visualizations in future
                                        if (v > a)
                                        {
                                            a = v;
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

        public override void DoPoolGradient(Volume<float> input, Volume<float> outputGradient,
            Volume<float> inputGradient, int windowWidth, int windowHeight,
            int horizontalPad, int verticalPad, int horizontalStride, int verticalStride)
        {
            var inputWidth = input.Shape.GetDimension(0);
            var inputHeight = input.Shape.GetDimension(1);

            var outputWidth = outputGradient.Shape.GetDimension(0);
            var outputHeight = outputGradient.Shape.GetDimension(1);
            var outputDepth = outputGradient.Shape.GetDimension(2);
            var batchSize = outputGradient.Shape.GetDimension(3);

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
                                    if (oy >= 0 && oy < inputHeight && ox >= 0 && ox < inputWidth)
                                    {
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
                            }

                            var chainGradient = outputGradient.Get(ax, ay, depth, n);
                            inputGradient.Storage.Set(winx, winy, depth, n, chainGradient);
                        }
                    }
                }
            }
        }

        public override void DoRelu(Volume<float> volume)
        {
            this.Storage.Map(x => x <= 0 ? 0 : x, volume.Storage);
        }

        public override void DoReluGradient(Volume<float> input, Volume<float> output, Volume<float> outputGradient)
        {
            this.Storage.Map((x, y) => x > 0 ? y : 0, output.Storage, outputGradient.Storage);
        }

        public override void DoSigmoid(Volume<float> volume)
        {
            this.Storage.Map(x => (float)(1.0 / (1.0 + Math.Exp(-x))), volume.Storage);
        }

        public override void DoSigmoidGradient(Volume<float> input, Volume<float> outputGradient, Volume<float> inputGradient)
        {
            this.Storage.Map((output, outGradient) => output * (1.0f - output) * outGradient, outputGradient.Storage, inputGradient.Storage);
        }

        public override void DoSoftMax(Volume<float> result)
        {
            var batchSize = this.Shape.GetDimension(3);

            var outputWidth = this.Shape.GetDimension(0);
            var outputHeight = this.Shape.GetDimension(1);
            var outputDepth = this.Shape.GetDimension(2);

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
                            var v = Get(ax, ay, depth, n);
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
                            var e = Math.Exp(Get(ax, ay, depth, n) - amax);
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

        public override void DoSoftMaxGradient(Volume<float> outputGradient, Volume<float> inputGradient)
        {
            this.Storage.Map((input, outputG) => (outputG - 1) * input + input, outputGradient.Storage,
                inputGradient.Storage);
        }

        public override void DoTanh(Volume<float> volume)
        {
            this.Storage.Map(x => (float)Math.Tanh(x), volume.Storage);
        }

        public override void DoTanhGradient(Volume<float> input, Volume<float> outputGradient, Volume<float> inputGradient)
        {
            this.Storage.Map((output, outGradient) => (1.0f - output * output) * outGradient, outputGradient.Storage, inputGradient.Storage);
        }
    }
}