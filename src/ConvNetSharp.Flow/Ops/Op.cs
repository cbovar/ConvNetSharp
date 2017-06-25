using System;
using System.Collections.Generic;
using ConvNetSharp.Flow.Graph;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public abstract class Op<T> : IDisposable
        where T : struct, IEquatable<T>, IFormattable
    {
        public bool IsDirty { get; set; } = true;

        public Volume<T> Result { get; set; }

        public Op<T> Derivate { get; set; }

        public List<Op<T>> Parents { get; } = new List<Op<T>>();

        public List<Op<T>> Children { get; } = new List<Op<T>>();

        public abstract string Representation { get; }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public void Accept(IOpVisitor<T> visitor)
        {
            visitor.Visit(this);
        }

        public void AddParent(Op<T> parent)
        {
            if (!this.Parents.Contains(parent))
            {
                this.Parents.Add(parent);
            }

            if (!this.Children.Contains(this))
            {
                parent.Children.Add(this);
            }
        }

        public abstract void Differentiate();

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                this.Derivate?.Dispose();
            }
        }

        public abstract Volume<T> Evaluate(Session<T> session);

        public virtual Dictionary<string, object> GetData()
        {
            return new Dictionary<string, object>();
        }

        public static Op<T> operator +(Op<T> left, Op<T> right)
        {
            return new Add<T>(left, right);
        }

        public static Op<T> operator /(Op<T> left, Op<T> right)
        {
            return new Div<T>(left, right);
        }

        public static Op<T> operator *(Op<T> left, Op<T> right)
        {
            return new Mult<T>(left, right);
        }

        public static Op<T> operator -(Op<T> left, Op<T> right)
        {
            return new Add<T>(left, -right);
        }

        public static Op<T> operator -(Op<T> x)
        {
            return new Negate<T>(x);
        }

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

        public void RemoveParent(Op<T> parent)
        {
            this.Parents.Remove(parent);
            parent.Children.Remove(this);
        }

        public void SetDirty()
        {
            this.IsDirty = true;
            foreach (var child in this.Children)
            {
                child.SetDirty();
            }
        }
    }
}