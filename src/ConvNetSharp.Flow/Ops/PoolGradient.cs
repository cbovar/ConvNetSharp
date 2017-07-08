using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class PoolGradient<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Pool<T> _pool;

        public PoolGradient(Pool<T> pool, Op<T> derivate)
        {
            this._pool = pool;

            this.AddParent(pool);
            this.AddParent(derivate);
        }

        public override string Representation => "PoolGradient";

        public override void Differentiate()
        {
            throw new NotImplementedException();
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return this._pool.InputGradient;
            }
            this.IsDirty = false;

            this._pool.EvaluateGradient(session);
            return this._pool.InputGradient;
        }
    }
}