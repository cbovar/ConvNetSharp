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
        public Assign(ConvNetSharp<T> graph, Op<T> valueOp, Op<T> op) : base(graph)
        {
            if (!(valueOp is Variable<T>))
            {
                throw new ArgumentException("Assigned Op should be a Variable", nameof(valueOp));
            }

            this.AddParent(valueOp);
            this.AddParent(op);
        }

        public override string Representation => "->";

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

            this.Result = this.Parents[1].Evaluate(session);
            ((Variable<T>)this.Parents[0]).SetValue(this.Result);

            return base.Evaluate(session);
        }

        public override string ToString()
        {
            return $"({this.Parents[0]} <- {this.Parents[1]})";
        }
    }
}