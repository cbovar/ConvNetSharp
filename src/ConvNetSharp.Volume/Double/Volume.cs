using System;

namespace ConvNetSharp.Volume.Double;

public class Volume : Volume<double>
{
    internal Volume(double[] array, Shape shape) : this(new NcwhVolumeStorage<double>(array, shape))
    {
    }

    internal Volume(VolumeStorage<double> storage) : base(storage)
    {
    }

    public override void Activation(ActivationType type, Volume<double> volume)
    {
        switch (type)
        {
            case ActivationType.Sigmoid:
                this.Storage.Map(x => 1.0 / (1.0 + Math.Exp(-x)), volume.Storage);
                return;
            case ActivationType.Relu:
                this.Relu(volume);
                break;
            case ActivationType.Tanh:
                this.Storage.Map(Math.Tanh, volume.Storage);
                break;
            case ActivationType.ClippedRelu:
                throw new NotImplementedException();
        }
    }

    public override void Exp(Volume<double> result)
    {
        this.Storage.Map(Math.Exp, result.Storage);
    }

    public override void Log(Volume<double> result)
    {
        this.Storage.Map(x => Math.Log(x), result.Storage);
    }

    public override void Norm1(Volume<double> result)
    {
        var batchSize = this.Shape.Dimensions[3];
        var reshape = this.ReShape(-1, batchSize);

        var n = reshape.Shape.Dimensions[0];

        for (var i = 0; i < batchSize; i++)
        {
            var sum = 0.0;

            for (var j = 0; j < n; j++)
            {
                var d = reshape.Get(j, i);
                sum += Math.Abs(d);
            }

            result.Set([i], sum);
        }
    }

    public override void Power(Volume<double> power, Volume<double> result)
    {
        this.Storage.MapEx(Math.Pow, power.Storage, result.Storage);
    }

    public override void Sigmoid(Volume<double> volume)
    {
        this.Storage.Map(x => 1.0 / (1.0 + Math.Exp(-x)), volume.Storage);
    }

    public override void Softmax(Volume<double> result)
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
                            es[ax + ay * outputWidth + depth * outputWidth * outputHeight]);
                    }
                }
            }
        }
    }

    public override void Sqrt(Volume<double> result)
    {
        this.Storage.Map(Math.Sqrt, result.Storage);
    }

    public override void Tanh(Volume<double> volume)
    {
        this.Storage.Map(Math.Tanh, volume.Storage);
    }
}