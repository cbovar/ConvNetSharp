using System;
using System.Collections.Generic;
using ConvNetSharp.Flow.Graph;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public abstract class Op<T> : IDisposable
        where T : struct, IEquatable<T>, IFormattable
    {
        public Op<T> Derivate { get; set; }

        public List<Op<T>> Parents { get; } = new List<Op<T>>();

        public List<Op<T>> Children { get; } = new List<Op<T>>();

        public void RegisterDerivate(Op<T> d)
        {
            if (this.Derivate == null)
            {
                this.Derivate = d;
            }
            else
            {
                this.Derivate += d;
            }
        }

        public void AddParent(Op<T> parent)
        {
            if (!this.Parents.Contains(parent))
            {
                this.Parents.Add(parent);
                parent.Children.Add(this);
            }
        }

        public void RemoveParent(Op<T> parent)
        {
            this.Parents.Remove(parent);
            parent.Children.Remove(this);
        }

        protected long LastComputeStep { get; set; } = -1;

        public void Accept(IOpVisitor<T> visitor)
        {
            visitor.Visit(this);
        }

        public abstract void Differentiate();

        public abstract Volume<T> Evaluate(Session<T> session);

        public static Op<T> operator +(Op<T> left, Op<T> right)
        {
            var opAddition = new AddOp<T>(left, right);

            left.Children.Add(opAddition);
            right.Children.Add(opAddition);

            return opAddition;
        }

        public static Op<T> operator -(Op<T> left, Op<T> right)
        {
            var opAddition = new AddOp<T>(left, -right);

            left.Children.Add(opAddition);
            right.Children.Add(opAddition);

            return opAddition;
        }

        public static Op<T> operator *(Op<T> left, Op<T> right)
        {
            var opMultiply = new MultOp<T>(left, right);

            left.Children.Add(opMultiply);
            right.Children.Add(opMultiply);

            return opMultiply;
        }

        public static Op<T> operator /(Op<T> left, Op<T> right)
        {
            var opMultiply = new DivOp<T>(left, right);

            left.Children.Add(opMultiply);
            right.Children.Add(opMultiply);

            return opMultiply;
        }

        public static Op<T> operator -(Op<T> x)
        {
            var opNegate = new NegateOp<T>(x);
            x.Children.Add(opNegate);
            return opNegate;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                this.Derivate?.Dispose();
            }
        }

        public abstract string Representation { get; }
    }
}