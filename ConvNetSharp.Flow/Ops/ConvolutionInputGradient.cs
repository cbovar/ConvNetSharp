using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    internal class ConvolutionInputGradient<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Convolution<T> _convolution;

        public ConvolutionInputGradient(Convolution<T> convolution, Op<T> derivate)
        {
            this._convolution = convolution;

            AddParent(convolution);
            AddParent(derivate);
        }

        public override string Representation => "ConvolutionInputGradient";

        public override void Differentiate()
        {
            throw new NotImplementedException();
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            this._convolution.EvaluateGradient(session);
            return this._convolution.InputGradient;
        }
    }
}