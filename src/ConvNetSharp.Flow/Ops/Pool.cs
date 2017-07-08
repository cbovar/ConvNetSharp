using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Pool<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private long _lastGradientComputeStep = -1;
        private Shape _lastInputShape;

        public Pool(Dictionary<string, object> data)
        {
            this.HorizontalPad = int.Parse((string)data["HorizontalPad"]);
            this.VerticalPad = int.Parse((string)data["VerticalPad"]);
            this.HorizontalStride = int.Parse((string)data["HorizontalStride"]);
            this.VerticalStride = int.Parse((string)data["VerticalStride"]);
            this.Width = int.Parse((string)data["Width"]);
            this.Height = int.Parse((string)data["Height"]);
        }

        public override Dictionary<string, object> GetData()
        {
            var data = base.GetData();
            data["Width"] = this.Width;
            data["Height"] = this.Height;
            data["HorizontalStride"] = this.HorizontalStride;
            data["VerticalStride"] = this.VerticalStride;
            data["HorizontalPad"] = this.HorizontalPad;
            data["VerticalPad"] = this.VerticalPad;

            return data;
        }

        public Pool(Op<T> x, int width, int height, int horizontalPad, int verticalPad, int horizontalStride, int verticalStride)
        {
            this.Width = width;
            this.Height = height;
            this.HorizontalStride = horizontalStride;
            this.VerticalStride = verticalStride;
            this.HorizontalPad = horizontalPad;
            this.VerticalPad = verticalPad;
            AddParent(x);
        }

        public int Width { get; }

        public int Height { get; }

        public int HorizontalStride { get; set; }

        public int VerticalStride { get; set; }

        public int HorizontalPad { get; set; }

        public int VerticalPad { get; set; }

        public override string Representation => $"Pool {this.Width}x{this.Height}";

        public Volume<T> InputGradient { get; private set; }

        public override void Differentiate()
        {
            this.Parents[0].RegisterDerivate(new PoolGradient<T>(this, this.Derivate));
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return this.Result;
            }
            this.IsDirty = false;

            var x = this.Parents[0].Evaluate(session);

            if (this.Result == null || !Equals(this._lastInputShape, x.Shape))
            {
                this._lastInputShape = new Shape(x.Shape);

                var outputShape = new Shape(
                    (int)Math.Floor((x.Shape.GetDimension(0) + this.HorizontalPad * 2 - this.Width) / (double)this.HorizontalStride + 1),
                    (int)Math.Floor((x.Shape.GetDimension(1) + this.VerticalPad * 2 - this.Height) / (double)this.VerticalStride + 1),
                    x.Shape.GetDimension(2),
                    x.Shape.GetDimension(3)
                );

                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(outputShape);
            }

            x.DoPool(this.Result, this.Width, this.Height, this.HorizontalPad, this.VerticalPad, this.HorizontalStride, this.VerticalStride);
            return this.Result;
        }

        public void EvaluateGradient(Session<T> session)
        {
            if (this._lastGradientComputeStep == session.Step)
            {
                return;
            }
            this._lastGradientComputeStep = session.Step;

            var x = this.Parents[0].Evaluate(session);

            if (this.InputGradient == null || !Equals(x.Shape, this.InputGradient.Shape))
            {
                this.InputGradient = BuilderInstance<T>.Volume.SameAs(x.Shape);
            }

            this.InputGradient.Clear();

            this.Result.DoPoolGradient(x, this.Derivate.Evaluate(session), this.InputGradient, this.Width, this.Height, this.HorizontalPad, this.VerticalPad, this.HorizontalStride,
                this.VerticalStride);
        }
    }
}