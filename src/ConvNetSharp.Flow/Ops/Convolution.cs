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
        private readonly ConvNetSharp<T> _cns;
        private readonly Variable<T> _filter;
        private readonly Op<T> _x;
        private long _lastGradientComputeStep = -1;

        private Shape _lastInputShape;

        public Convolution(Op<T> x, int width, int height, int filterCount, int stride = 1, int pad = 0, ConvNetSharp<T> cns = null)
        {
            this.Stride = stride;
            this.Pad = pad;
            this.FilterCount = filterCount;
            this.Width = width;
            this.Height = height;

            this._x = x;
            AddParent(x);

            this._cns = cns ?? ConvNetSharp<T>.Instance;
            this._filter = this._cns.Variable(null, "Filter"); // dummy
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
            this._x.RegisterDerivate(new ConvolutionInputGradient<T>(this, this.Derivate));
            this._filter.RegisterDerivate(new ConvolutionFilterGradient<T>(this, this.Derivate));
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                this.Result?.Dispose();
                this._filter?.Dispose();
                this.FilterGradient?.Dispose();
                this.InputGradient?.Dispose();
            }

            base.Dispose(disposing);
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return this.Result;
            }
            this.IsDirty = false;

            var x = this._x.Evaluate(session);

            // Allocate result and filters if needed
            if (this.Result == null || !Equals(this._lastInputShape, x.Shape))
            {
                this._lastInputShape = new Shape(x.Shape);

                var count = this.Width * this.Height * x.Shape.GetDimension(2);
                var scale = Math.Sqrt(2.0 / count);

                this._filter.Result = BuilderInstance<T>.Volume.Random(new Shape(this.Width, this.Height, x.Shape.GetDimension(2), this.FilterCount), 0.0, scale);

                var outputDepth = this.FilterCount;
                var outputWidth = (int) Math.Floor((x.Shape.GetDimension(0) + this.Pad * 2 - this.Width) / (double) this.Stride + 1);
                var outputHeight = (int) Math.Floor((x.Shape.GetDimension(1) + this.Pad * 2 - this.Height) / (double) this.Stride + 1);

                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(new Shape(outputWidth, outputHeight, outputDepth, x.Shape.GetDimension(3)));
            }

            x.DoConvolution(this._filter.Evaluate(session), this.Pad, this.Stride, this.Result);

            return this.Result;
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

            this.FilterGradient.Clear();
            this.InputGradient.Clear();

            x.DoConvolutionGradient(filter, this.Derivate.Evaluate(session), this.InputGradient, this.FilterGradient, this.Pad, this.Stride);
        }

        public override string ToString()
        {
            return $"Conv({this._x})";
        }
    }
}