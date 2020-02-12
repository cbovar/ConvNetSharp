using System;
using System.Collections.Generic;
using ConvNetSharp.Core;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     Describes the shape of the data.
    ///     Shape always has 4 dimensions: [width, height, class, batch size]
    ///     e.g. A 1D array fits in a volume that has a shape of [1,1,n,1]
    ///     A 2D array fits in a volume that has a shape of [w,h,1,1]
    ///     A 2D array with 3 channels (a color image for example) fits in a volume that has a shape of [w,h,3,1]
    ///     10 2D arrays (e.g. 10 grayscale images) fits in a volume that a shape of [w,h,1,10]
    /// </summary>
    /// <typeparam name="T">type of data (double or float)</typeparam>
    public class Shape<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly VolumeBuilder<T> _builder;

        public Shape(ConvNetSharp<T> graph, Op<T> x, int index) : this(graph, x)
        {
            this.Index = index;
        }

        public Shape(ConvNetSharp<T> graph, Op<T> x) : base(graph)
        {
            this._builder = BuilderInstance<T>.Create(); // we want to remain on host
            this.AddParent(x);
        }

        public Shape(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
            this._builder = BuilderInstance<T>.Create(); // we want to remain on host
            this.Index = int.Parse((string) data["index"]);
        }

        public int Index { get; } = -1;

        public override string Representation => "Shape";

        public override void Differentiate()
        {
            throw new NotImplementedException();
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return base.Evaluate(session);
            }

            this.IsDirty = false;

            var y = this.Parents[0].Evaluate(session);

            if (this.Result == null)
            {
                this.Result = this._builder.SameAs(this.Index == -1 ? new Shape(4) : new Shape(1));
            }

            if (this.Index == -1)
            {
                this.Result.Set(0, Ops<T>.Cast(y.Shape.Dimensions[0]));
                this.Result.Set(1, Ops<T>.Cast(y.Shape.Dimensions[1]));
                this.Result.Set(2, Ops<T>.Cast(y.Shape.Dimensions[2]));
                this.Result.Set(3, Ops<T>.Cast(y.Shape.Dimensions[3]));
            }
            else
            {
                this.Result.Set(0, Ops<T>.Cast(y.Shape.Dimensions[this.Index]));
            }

            return base.Evaluate(session);
        }

        public override Dictionary<string, object> GetData()
        {
            var data = base.GetData();
            data["index"] = this.Index;
            return data;
        }

        public override string ToString()
        {
            return $"shape({this.Parents[0]})";
        }
    }
}