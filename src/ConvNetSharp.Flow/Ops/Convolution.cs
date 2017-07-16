using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     y = conv(x)
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Convolution<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Variable<T> _filter;
        private long _lastGradientComputeStep = -1;
        private Shape _lastInputShape;

        public Convolution(Dictionary<string, object> data)
        {
            this.Stride = int.Parse((string) data["Stride"]);
            this.Pad = int.Parse((string) data["Pad"]);
            this.FilterCount = int.Parse((string) data["FilterCount"]);
            this.Width = int.Parse((string) data["Width"]);
            this.Height = int.Parse((string) data["Height"]);
        }

        public Convolution(Op<T> x, int width, int height, int filterCount, int stride = 1, int pad = 0, ConvNetSharp<T> cns = null)
        {
            this.Stride = stride;
            this.Pad = pad;
            this.FilterCount = filterCount;
            this.Width = width;
            this.Height = height;

            AddParent(x);

            var _cns = cns ?? ConvNetSharp<T>.Instance;
            this._filter = _cns.Variable($"Filter_{Count}"); // dummy
            AddParent(this._filter);
        }

        public Op<T> Filter => this._filter;

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
            this.Parents[0].RegisterDerivate(new ConvolutionInputGradient<T>(this, this.Derivate));
            this.Parents[1].RegisterDerivate(new ConvolutionFilterGradient<T>(this, this.Derivate));
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                this.Result?.Dispose();
                this.Parents[1]?.Dispose();
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

            var x = this.Parents[0].Evaluate(session);

            // Allocate result and filters if needed
            if (this.Result == null || !Equals(this._lastInputShape, x.Shape))
            {
                this._lastInputShape = new Shape(x.Shape);

                if (this.Parents[1].Result == null)
                {
                    var count = this.Width * this.Height * x.Shape.GetDimension(2);
                    var scale = Math.Sqrt(2.0 / count);

                    var filterShape = new Shape(this.Width, this.Height, x.Shape.GetDimension(2), this.FilterCount);
                    this.Parents[1].Result = BuilderInstance<T>.Volume.Random(filterShape, 0.0, scale);
                }

                var outputDepth = this.FilterCount;
                var outputWidth = (int) Math.Floor((x.Shape.GetDimension(0) + this.Pad * 2 - this.Width) / (double) this.Stride + 1);
                var outputHeight = (int) Math.Floor((x.Shape.GetDimension(1) + this.Pad * 2 - this.Height) / (double) this.Stride + 1);

                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(new Shape(outputWidth, outputHeight, outputDepth, x.Shape.GetDimension(3)));
            }

            x.DoConvolution(this.Parents[1].Evaluate(session), this.Pad, this.Stride, this.Result);

            return this.Result;
        }

        public void EvaluateGradient(Session<T> session)
        {
            if (this._lastGradientComputeStep == session.Step)
            {
                return;
            }
            this._lastGradientComputeStep = session.Step;

            var filter = this.Parents[1].Evaluate(session);

            if (this.FilterGradient == null || !Equals(filter.Shape, this.FilterGradient.Shape))
            {
                this.FilterGradient?.Dispose();
                this.FilterGradient = BuilderInstance<T>.Volume.SameAs(filter.Shape);
            }

            var x = this.Parents[0].Evaluate(session);

            if (this.InputGradient == null || !Equals(x.Shape, this.InputGradient.Shape))
            {
                this.InputGradient?.Dispose();
                this.InputGradient = BuilderInstance<T>.Volume.SameAs(x.Shape);
            }

            this.FilterGradient.Clear();
            this.InputGradient.Clear();

            var outputGradients = this.Derivate.Evaluate(session);
            x.DoConvolutionGradient(filter, outputGradients, this.InputGradient, this.FilterGradient, this.Pad, this.Stride);
        }

        public override Dictionary<string, object> GetData()
        {
            var data = base.GetData();

            data["Stride"] = this.Stride;
            data["Pad"] = this.Pad;
            data["FilterCount"] = this.FilterCount;
            data["Width"] = this.Width;
            data["Height"] = this.Height;

            return data;
        }

        public override string ToString()
        {
            return $"Conv({this.Parents[0]})";
        }
    }
}