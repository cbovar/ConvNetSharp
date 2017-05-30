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

        public override void DoConvolution(Volume<double> filters, int pad, int stride, Volume<double> result)
        {
        }

        public override void DoConvolutionGradient(Volume<double> filters, Volume<double> outputGradients, Volume<double> inputGradient, Volume<double> filterGradient, int pad, int stride)
        {
            throw new System.NotImplementedException();
        }

        public override void DoMultiply(Volume<double> other, Volume<double> result)
        {
            this.DoMultiplyCount++;
        }

        public override void DoNegate(Volume<double> result)
        {
            this.DoNegateCount++;
        }

        public override void DoSoftmax(Volume<double> result)
        {
            throw new NotImplementedException();
        }

        public override void DoSoftmaxGradient(Volume<double> outputGradient, Volume<double> result)
        {
            throw new System.NotImplementedException();
        }
    }
}