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

        public override void DoActivation(ActivationType type, Volume<double> result)
        {
        }

        public override void DoActivationGradient(Volume<double> input, Volume<double> outputGradient, ActivationType type, Volume<double> result)
        {
        }

        public override void DoAdd(Volume<double> other, Volume<double> result)
        {
            this.DoAddCount++;
        }

        public override void DoAdd(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoBiasGradient(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoConcat(Volume<double> right, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoConvolution(Volume<double> filters, int pad, int stride, Volume<double> result)
        {
        }

        public override void DoConvolutionGradient(Volume<double> filters, Volume<double> outputGradients, Volume<double> filterGradient, int pad,
            int stride, Volume<double> inputGradient)
        {
            throw new NotImplementedException();
        }

        public override void DoDivide(Volume<double> right, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoDropout(double dropProbability, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoDropoutGradient(Volume<double> input, Volume<double> outputGradient, Volume<double> inputGradient, double dropProbability)
        {
            throw new NotImplementedException();
        }

        public override void DoExp(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoExtract(int length, int offset, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoLeakyRelu(double alpha, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoLeakyReluGradient(Volume<double> outputGradient, Volume<double> inputGradient, double alpha)
        {
            throw new NotImplementedException();
        }

        public override void DoLog(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoPower(Volume<double> v, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoReduce(TensorReduceOp op, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoMax(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoMin(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoMultiply(double factor, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoMultiply(Volume<double> other, Volume<double> result)
        {
            this.DoMultiplyCount++;
        }

        public override void DoNegate(Volume<double> result)
        {
            this.DoNegateCount++;
        }

        public override void DoPool(int windowWidth, int windowHeight, int horizontalPad, int verticalPad, int horizontalStride, int verticalStride, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoPoolGradient(Volume<double> input, Volume<double> outputGradient, int windowWidth, int windowHeight, int horizontalPad,
            int verticalPad, int horizontalStride, int verticalStride, Volume<double> inputGradient)
        {
            throw new NotImplementedException();
        }

        public override void DoRelu(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoReluGradient(Volume<double> input, Volume<double> outputGradient, Volume<double> inputGradient)
        {
            throw new NotImplementedException();
        }

        public override void DoSigmoid(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoSigmoidGradient(Volume<double> input, Volume<double> outputGradient, Volume<double> inputGradient)
        {
            throw new NotImplementedException();
        }

        public override void DoSoftmax(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoSoftmaxGradient(Volume<double> outputGradient, Volume<double> inputGradient)
        {
            throw new NotImplementedException();
        }

        public override void DoSqrt(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoSubtractFrom(Volume<double> other, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoSum(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoNorm1(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoTanh(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoTanhGradient(Volume<double> input, Volume<double> outputGradient, Volume<double> inputGradient)
        {
            throw new NotImplementedException();
        }

        public override void DoTile(Volume<double> reps, Volume<double> result)
        {
            throw new NotImplementedException();
        }
    }
}