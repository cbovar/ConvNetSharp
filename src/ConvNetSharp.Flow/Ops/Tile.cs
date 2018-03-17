using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    /// Construct a volume by repeating x of times given by reps
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Tile<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private int _lastBatchSize;
        private Shape _tempShape;


        public Tile(Op<T> x, Op<T> reps)
        {
            AddParent(x);
            AddParent(reps);
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
                return this.Result;
            }
            this.IsDirty = false;

            var x = this.Parents[0].Evaluate(session);
            var reps = this.Parents[1].Evaluate(session);

            if (this._tempShape == null || session.BatchSize != this._lastBatchSize)
            {
                var dim0 = x.Shape.GetDimension(0) * Convert.ToInt32(reps.Get(0));
                var dim1 = x.Shape.GetDimension(1) * Convert.ToInt32(reps.Get(1));
                var dim2 = x.Shape.GetDimension(2) * Convert.ToInt32(reps.Get(2));
                var dim3 = x.Shape.GetDimension(3);

                this._tempShape = new Shape(dim0, dim1, dim2, dim3);
                this._lastBatchSize = session.BatchSize;
                this.Result = BuilderInstance<T>.Volume.SameAs(this._tempShape);
            }

            this.Result.Clear();

            x.DoTile(reps, this.Result);

            return this.Result;
        }
    }
}