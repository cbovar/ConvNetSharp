using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     Assignment: valueOp = op
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Assign<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private long _lastComputeStep;

        public Assign(Op<T> valueOp, Op<T> op)
        {
            if (!(valueOp is IValueOp<T>))
            {
                throw new ArgumentException("Assigned Op should implement IValueOp interface", nameof(valueOp));
            }

            AddParent(valueOp);
            AddParent(op);
        }

        public override string Representation => "->";

        public override void Differentiate()
        {
            throw new NotImplementedException();
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (this._lastComputeStep == session.Step)
            {
                return this.Parents[0].Evaluate(session);
            }
            this._lastComputeStep = session.Step;

            var op = this.Parents[1].Evaluate(session);
            ((IValueOp<T>)this.Parents[0]).SetValue(op);

            return op;
        }
    }
}