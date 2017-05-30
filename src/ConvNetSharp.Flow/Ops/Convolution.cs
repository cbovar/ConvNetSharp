using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     y = conv(x)
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Convolution<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Op<T> _x;
        private Variable<T> _filter;
        private long _lastGradientComputeStep = -1;
        private Shape _lastInputShape;
        private Volume<T> _result;

        public Convolution(Op<T> x, int width, int height, int filterCount, int stride = 1, int pad = 0)
        {
            this.Stride = stride;
            this.Pad = pad;
            this.FilterCount = filterCount;
            this.Width = width;
            this.Height = height;

            this._x = x;
            AddParent(x);

            this._filter = new Variable<T>(null, "Filter"); // dummy
            AddParent(this._filter);
        }

        public int Stride { get; set; }

        public int Pad { get; set; }

        public int FilterCount { get; }

        public int Width { get; }

        public int Height { get; }

        public override string Representation => this.Width == 1 && this.Height == 1 ? "FullyCon" : $"Conv {this.Width}x{this.Height}x{this.FilterCount}";

        public Volume<T> FilterGradient { get; private set; }

        public Volume<T> InputGradient { get; private set; }

        public override void Differentiate()
        {
            this._x.Derivate = new ConvolutionInputGradient<T>(this, this.Derivate);
            this._filter.Derivate = new ConvolutionFilterGradient<T>(this, this.Derivate);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                this._result?.Dispose();
                this._filter?.Dispose();
                this.FilterGradient?.Dispose();
                this.InputGradient?.Dispose();
            }

            base.Dispose(disposing);
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            // Do not compute when already computed
            if (this.LastComputeStep == session.Step)
            {
                return this._result;
            }
            this.LastComputeStep = session.Step;

            var x = this._x.Evaluate(session);

            // Allocate result and filters if needed
            if (this._result == null || !Equals(this._lastInputShape, x.Shape))
            {
                this._lastInputShape = new Shape(x.Shape);

                if (this._filter != null)
                {
                    RemoveParent(this._filter);
                    this._filter.Dispose();
                }
                this._filter = new Variable<T>(BuilderInstance<T>.Volume.SameAs(new Shape(this.Width, this.Height, x.Shape.GetDimension(2), this.FilterCount)), "Filter");

                AddParent(this._filter);

                var outputDepth = this.FilterCount;
                var outputWidth = (int) Math.Floor((x.Shape.GetDimension(0) + this.Pad * 2 - this.Width) / (double) this.Stride + 1);
                var outputHeight = (int) Math.Floor((x.Shape.GetDimension(1) + this.Pad * 2 - this.Height) / (double) this.Stride + 1);

                this._result?.Dispose();
                this._result = BuilderInstance<T>.Volume.SameAs(new Shape(outputWidth, outputHeight, outputDepth, x.Shape.GetDimension(4)));
            }

            x.DoConvolution(this._filter.Evaluate(session), this.Pad, this.Stride, this._result);

            return this._result;
        }

        public void EvaluateGradient(Session<T> session)
        {
            if (this._lastGradientComputeStep == session.Step)
            {
                return;
            }
            this._lastGradientComputeStep = session.Step;

            var filter = this._filter.Evaluate(session);

            if (this.FilterGradient == null || !Equals(filter.Shape, this.FilterGradient.Shape))
            {
                this.FilterGradient = BuilderInstance<T>.Volume.SameAs(filter.Shape);
            }

            var x = this._x.Evaluate(session);

            if (this.InputGradient == null || !Equals(x.Shape, this.InputGradient.Shape))
            {
                this.InputGradient = BuilderInstance<T>.Volume.SameAs(x.Shape);
            }

            x.DoConvolutionGradient(filter, this.Derivate.Evaluate(session), this.InputGradient, this.FilterGradient, this.Pad, this.Stride);
        }
    }
}