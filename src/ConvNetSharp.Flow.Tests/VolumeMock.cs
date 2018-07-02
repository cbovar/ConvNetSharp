using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Tests
{
    public class VolumeMock : Volume<double>
    {
        public VolumeMock(double value, Shape shape) : base(new NcwhVolumeStorage<double>(new[] { value }, shape))
        {
        }

        public int DoAddCount { get; set; }

        public int DoMultiplyCount { get; set; }

        public int DoNegateCount { get; set; }

        public override void Activation(ActivationType type, Volume<double> result)
        {
        }

        public override void ActivationGradient(Volume<double> input, Volume<double> outputGradient, ActivationType type, Volume<double> result)
        {
        }

        public override void Add(Volume<double> other, Volume<double> result)
        {
            this.DoAddCount++;
        }

        public override void Add(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void BiasGradient(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void Concat(Volume<double> right, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void Convolution(Volume<double> filters, int pad, int stride, Volume<double> result)
        {
        }

        public override void ConvolutionGradient(Volume<double> filters, Volume<double> outputGradients, Volume<double> filterGradient, int pad,
            int stride, Volume<double> inputGradient)
        {
            throw new NotImplementedException();
        }

        public override void Divide(Volume<double> right, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void Dropout(double dropProbability, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DropoutGradient(Volume<double> input, Volume<double> outputGradient, double dropProbability, Volume<double> inputGradient)
        {
            throw new NotImplementedException();
        }

        public override void Exp(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void Extract(int length, int offset, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void LeakyRelu(double alpha, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void LeakyReluGradient(Volume<double> outputGradient, Volume<double> inputGradient, double alpha)
        {
            throw new NotImplementedException();
        }

        public override void Log(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void Max(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void Min(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void Multiply(double factor, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void Multiply(Volume<double> other, Volume<double> result)
        {
            this.DoMultiplyCount++;
        }

        public override void MatMultiply(Volume<double> right, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void Negate(Volume<double> result)
        {
            this.DoNegateCount++;
        }

        public override void Norm1(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void Transpose(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void Pool(int windowWidth, int windowHeight, int horizontalPad, int verticalPad, int horizontalStride, int verticalStride, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void PoolGradient(Volume<double> input, Volume<double> outputGradient, int windowWidth, int windowHeight, int horizontalPad,
            int verticalPad, int horizontalStride, int verticalStride, Volume<double> inputGradient)
        {
            throw new NotImplementedException();
        }

        public override void Power(Volume<double> v, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void Reduce(TensorReduceOp op, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void Relu(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void ReluGradient(Volume<double> input, Volume<double> outputGradient, Volume<double> inputGradient)
        {
            throw new NotImplementedException();
        }

        public override void Sigmoid(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void SigmoidGradient(Volume<double> input, Volume<double> outputGradient, Volume<double> inputGradient)
        {
            throw new NotImplementedException();
        }

        public override void Softmax(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void SoftmaxGradient(Volume<double> outputGradient, Volume<double> inputGradient)
        {
            throw new NotImplementedException();
        }

        public override void Sqrt(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void SubtractFrom(Volume<double> other, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void Sum(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void Tanh(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void TanhGradient(Volume<double> input, Volume<double> outputGradient, Volume<double> inputGradient)
        {
            throw new NotImplementedException();
        }

        public override void Tile(Volume<double> reps, Volume<double> result)
        {
            throw new NotImplementedException();
        }
    }
}