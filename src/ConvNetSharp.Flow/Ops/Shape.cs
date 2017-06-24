using System;
using ConvNetSharp.Core;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Shape<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Op<T> _x;
        private readonly VolumeBuilder<T> _builder;
        public override string Representation => "Shape";

        public Shape(Op<T> x)
        {
            this._builder = BuilderInstance<T>.Create(); // we want to remain on host

            this._x = x;
            AddParent(x);
        }

        public override void Differentiate()
        {
            throw new NotImplementedException();
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return this.Result;
            }
            this.IsDirty = false;

            var y = this._x.Evaluate(session);

            if (this.Result == null)
            {
                this.Result = this._builder.SameAs(new Shape(4));
            }

            this.Result.Set(0, Ops<T>.Cast(y.Shape.GetDimension(0)));
            this.Result.Set(1, Ops<T>.Cast(y.Shape.GetDimension(1)));
            this.Result.Set(2, Ops<T>.Cast(y.Shape.GetDimension(2)));
            this.Result.Set(3, Ops<T>.Cast(y.Shape.GetDimension(3)));

            return this.Result;
        }
    }
}