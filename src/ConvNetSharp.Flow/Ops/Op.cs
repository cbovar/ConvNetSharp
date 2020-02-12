using System;
using System.Collections.Generic;
using ConvNetSharp.Flow.Graph;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public abstract class Op<T> : IDisposable
        where T : struct, IEquatable<T>, IFormattable
    {
        public ConvNetSharp<T> Graph;

        protected Op(ConvNetSharp<T> cns)
        {
            this.Graph = cns;
            Count++;
        }

        public static int Count { get; set; } = 1;

        public bool IsDirty { get; set; } = true;

        public Volume<T> Result { get; set; }

        public Op<T> Derivate { get; set; }

        public List<Op<T>> Parents { get; } = new List<Op<T>>();

        public List<Op<T>> Children { get; } = new List<Op<T>>();

        public virtual string Representation { get; }

        public void Dispose()
        {
            this.Dispose(true);
            GC.SuppressFinalize(this);
        }

        public void Accept(IOpVisitor<T> visitor)
        {
            visitor.Visit(this);
        }

        public void AddParent(Op<T> parent)
        {
            this.Parents.Add(parent);

            if (!this.Children.Contains(this))
            {
                parent.Children.Add(this);
            }
        }

        public virtual void Differentiate()
        {
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                this.Derivate?.Dispose();
            }
        }

        public static void DisposeGraph(Op<T> root)
        {
            var visitor = new OpVisitor<T>(op =>
            {
                if (!(op is INamedOp<T>)) // not sure about this
                {
                    op.Dispose();
                }
            }, true);
            root?.Accept(visitor);
        }

        public virtual Volume<T> Evaluate(Session<T> session)
        {
            this.Evaluated?.Invoke(this, new EventArgs());

#if DEBUG
            var inputs = this.Result.ToArray();
            for (var index = 0; index < inputs.Length; index++)
            {
                var i = inputs[index];
                if (Core.Ops<T>.IsInvalid(i))
                {
                    throw new ArgumentException("Invalid input!");
                }
            }
#endif

            return this.Result;
        }

        public event EventHandler Evaluated;

        ~Op()
        {
            this.Dispose(false);
        }

        public virtual Dictionary<string, object> GetData()
        {
            return new Dictionary<string, object>();
        }

        public static Op<T> operator +(Op<T> left, Op<T> right)
        {
            if (left.Graph != right.Graph)
            {
                throw new Exception("Graph are different");
            }

            return new Add<T>(left.Graph, left, right);
        }

        public static Op<T> operator /(Op<T> left, Op<T> right)
        {
            if (left.Graph != right.Graph)
            {
                throw new Exception("Graph are different");
            }

            return new Div<T>(left.Graph, left, right);
        }

        public static Op<T> operator *(Op<T> left, Op<T> right)
        {
            if (left.Graph != right.Graph)
            {
                throw new Exception("Graph are different");
            }

            return new Mult<T>(left.Graph, left, right);
        }

        public static Op<T> operator ^(Op<T> left, Op<T> right)
        {
            if (left.Graph != right.Graph)
            {
                throw new Exception("Graph are different");
            }

            return new Power<T>(left.Graph, left, right);
        }

        public static Op<T> operator -(Op<T> left, Op<T> right)
        {
            if (left.Graph != right.Graph)
            {
                throw new Exception("Graph are different");
            }

            return new Add<T>(left.Graph, left, -right);
        }

        public static Op<T> operator -(Op<T> x)
        {
            return new Negate<T>(x.Graph, x);
        }

        public static implicit operator Op<T>(T x)
        {
            return new Const<T>(ConvNetSharp<T>.Default, x, x.ToString()); // Use Default graph => can we do better ?
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
                if (!child.IsDirty)
                {
                    child.SetDirty();
                }
            }
        }
    }
}