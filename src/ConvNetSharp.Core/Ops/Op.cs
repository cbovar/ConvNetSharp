using System;
using System.Collections.Generic;
using ConvNetSharp.Core.Graph;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Ops
{
    public abstract class Op<T> : IDisposable
        where T : struct, IEquatable<T>, IFormattable
    {
        public Op<T> Derivate { get; set; }

        public List<Op<T>> Parents { get; } = new List<Op<T>>();

        public List<Op<T>> Children { get; } = new List<Op<T>>();

        public void Accept(IOpVisitor<T> visitor)
        {
            visitor.Visit(this);
        }

        public abstract void Backward();

        public abstract Volume<T> Forward(Session<T> session);

        public static Op<T> operator +(Op<T> left, Op<T> right)
        {
            var opAddition = new AddOp<T> { Parents = { left, right } };

            left.Children.Add(opAddition);
            right.Children.Add(opAddition);

            return opAddition;
        }

        public static Op<T> operator -(Op<T> left, Op<T> right)
        {
            var opAddition = new AddOp<T> { Parents = { left, -right } };

            left.Children.Add(opAddition);
            right.Children.Add(opAddition);

            return opAddition;
        }

        public static Op<T> operator *(Op<T> left, Op<T> right)
        {
            var opMultiply = new MultOp<T> { Parents = { left, right } };

            left.Children.Add(opMultiply);
            right.Children.Add(opMultiply);

            return opMultiply;
        }

        public static Op<T> operator -(Op<T> x)
        {
            var opNegate = new NegateOp<T> { Parents = { x } };
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