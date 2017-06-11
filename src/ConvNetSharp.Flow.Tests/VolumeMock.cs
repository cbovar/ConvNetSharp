using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Tests
{
    public class VolumeMock : Volume<double>
    {
        public VolumeMock(double value, Shape shape) : base(new NcwhVolumeStorage<double>(new[] {value}, shape))
        {
        }

        public int DoAddCount { get; set; }

        public int DoMultiplyCount { get; set; }

        public int DoNegateCount { get; set; }

        public override void DoActivation(Volume<double> result, ActivationType type)
        {
        }

        public override void DoActivationGradient(Volume<double> input, Volume<double> outputGradient, Volume<double> result, ActivationType type)
        {
        }

        public override void DoAdd(Volume<double> other, Volume<double> result)
        {
            this.DoAddCount++;
        }

        protected override void DoBiasGradient(Volume<double> biasGradient)
        {
            throw new NotImplementedException();
        }

        public override void DoConvolution(Volume<double> filters, int pad, int stride, Volume<double> result)
        {
        }

        public override void DoConvolutionGradient(Volume<double> filters, Volume<double> outputGradients, Volume<double> inputGradient, Volume<double> filterGradient, int pad,
            int stride)
        {
            throw new NotImplementedException();
        }

        public override void DoDivide(Volume<double> right, Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoExp(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoLog(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoMax(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoMultiply(Volume<double> result, double factor)
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

        public override void DoPool(Volume<double> result, int windowWidth, int windowHeight, int horizontalPad, int verticalPad, int horizontalStride, int verticalStride)
        {
            throw new NotImplementedException();
        }

        public override void DoPoolGradient(Volume<double> input, Volume<double> outputGradient, Volume<double> inputGradient, int windowWidth, int windowHeight, int horizontalPad,
            int verticalPad, int horizontalStride,
            int verticalStride)
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

        public override void DoSoftMax(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoSoftMaxGradient(Volume<double> outputGradient, Volume<double> inputGradient)
        {
            throw new NotImplementedException();
        }

        public override void DoSubtractFrom(Volume<double> other, Volume<double> result)
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
    }
}