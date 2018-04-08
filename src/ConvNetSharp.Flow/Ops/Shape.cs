using System;
using System.Collections.Generic;
using ConvNetSharp.Core;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
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
            AddParent(x);
        }

        public Shape(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
            this._builder = BuilderInstance<T>.Create(); // we want to remain on host
            this.Index = int.Parse((string)data["index"]);
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
                this.Result.Set(0, Ops<T>.Cast(y.Shape.GetDimension(0)));
                this.Result.Set(1, Ops<T>.Cast(y.Shape.GetDimension(1)));
                this.Result.Set(2, Ops<T>.Cast(y.Shape.GetDimension(2)));
                this.Result.Set(3, Ops<T>.Cast(y.Shape.GetDimension(3)));
            }
            else
            {
                this.Result.Set(0, Ops<T>.Cast(y.Shape.GetDimension(this.Index)));
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