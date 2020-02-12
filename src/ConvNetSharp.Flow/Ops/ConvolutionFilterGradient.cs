using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    internal class ConvolutionFilterGradient<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Convolution<T> _convolution;

        public ConvolutionFilterGradient(ConvNetSharp<T> graph, Convolution<T> convolution, Op<T> derivate) : base(graph)
        {
            this._convolution = convolution;

            this.AddParent(convolution);
            this.AddParent(derivate);
        }

        public override string Representation => "ConvolutionFilterGradient";

        public override void Differentiate()
        {
            throw new NotImplementedException();
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return this._convolution.FilterGradient;
            }

            this.IsDirty = false;

            this._convolution.EvaluateGradient(session);
            return this._convolution.FilterGradient;
        }
    }
}