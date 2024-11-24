using System;

namespace ConvNetSharp.Volume.Single;

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

    public override void Exp(Volume<float> result)
    {
        this.Storage.Map(x => (float)Math.Exp(x), result.Storage);
    }

    public override void Log(Volume<float> result)
    {
        this.Storage.Map(x => (float)Math.Log(x), result.Storage);
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

            result.Set([i], sum);
        }
    }

    public override void Power(Volume<float> power, Volume<float> result)
    {
        this.Storage.MapEx((x, y) => (float)Math.Pow(x, y), power.Storage, result.Storage);
    }

    public override void Sigmoid(Volume<float> volume)
    {
        this.Storage.Map(x => (float)(1.0 / (1.0 + Math.Exp(-x))), volume.Storage);
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

    public override void Sqrt(Volume<float> result)
    {
        this.Storage.Map(x => (float)Math.Sqrt(x), result.Storage);
    }

    public override void Tanh(Volume<float> volume)
    {
        this.Storage.Map(x => (float)Math.Tanh(x), volume.Storage);
    }
}