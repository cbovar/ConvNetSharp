using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     Construct a volume by repeating x of times given by reps
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Tile<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private int _lastBatchSize;
        private Shape _tempShape;


        public Tile(ConvNetSharp<T> graph, Op<T> x, Op<T> reps) : base(graph)
        {
            this.AddParent(x);
            this.AddParent(reps);
        }

        public Tile(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
        }

        public override string Representation => "Tile";

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

            var x = this.Parents[0].Evaluate(session);
            var reps = this.Parents[1].Evaluate(session);

            if (this._tempShape == null || session.BatchSize != this._lastBatchSize)
            {
                var dim0 = Math.Max(x.Shape.Dimensions[0] * Convert.ToInt32(reps.Get(0)), 1);
                var dim1 = Math.Max(x.Shape.Dimensions[1] * Convert.ToInt32(reps.Get(1)), 1);
                var dim2 = Math.Max(x.Shape.Dimensions[2] * Convert.ToInt32(reps.Get(2)), 1);
                var dim3 = Math.Max(x.Shape.Dimensions[3] * Convert.ToInt32(reps.Get(3)), 1);

                this._tempShape = new Shape(dim0, dim1, dim2, dim3);
                this._lastBatchSize = session.BatchSize;
                this.Result = BuilderInstance<T>.Volume.SameAs(this._tempShape);
            }

            this.Result.Clear();

            x.Tile(reps, this.Result);

            return base.Evaluate(session);
        }
    }
}