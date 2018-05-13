using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Extract<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public override string Representation => "Extract";

        private int lastTotalLength = 0;

        public Extract(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
        }

        public Extract(ConvNetSharp<T> graph, Op<T> x, Op<T> length, Op<T> offset) : base(graph)
        {
            AddParent(x);
            AddParent(length);
            AddParent(offset);
        }

        public override string ToString()
        {
            return $"Extract({this.Parents[0]}, {this.Parents[1]}, {this.Parents[2]})";
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return base.Evaluate(session);
            }
            this.IsDirty = false;

            var x = this.Parents[0].Evaluate(session);
            var length = (int)Convert.ChangeType(this.Parents[1].Evaluate(session).Get(0), typeof(int)); // TODO: Find a way to keep this on host
            var offset = (int)Convert.ChangeType(this.Parents[2].Evaluate(session).Get(0), typeof(int)); // TODO: Find a way to keep this on host

            var batchSize = x.Shape.Dimensions[3];

            int totalLength = length * batchSize;
            if (this.Result == null || this.lastTotalLength != totalLength)
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, length, batchSize));
            }

            var isScalar = x.Shape.TotalLength == 1;

            if (isScalar)
            {
                x.DoTile(this.Result.Shape.ToVolume<T>(), this.Result);
            }
            else
            {
                x.DoExtract(length, offset, this.Result);
            }

            return base.Evaluate(session);
        }
    }
}