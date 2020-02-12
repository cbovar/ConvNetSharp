using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Concat<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly int lastTotalLength = 0;

        public Concat(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
        }

        public Concat(ConvNetSharp<T> graph, Op<T> left, Op<T> right) : base(graph)
        {
            this.AddParent(left);
            this.AddParent(right);
        }

        public override string Representation => "Concat";

        public override void Differentiate()
        {
            var flattenShape = new Shape(1, 1, -1, Shape.Keep);
            var lengthLeft = new Shape<T>(this.Graph, new Reshape<T>(this.Graph, this.Parents[0], flattenShape), 2);
            var lengthRight = new Shape<T>(this.Graph, new Reshape<T>(this.Graph, this.Parents[1], flattenShape), 2);

            var extractLeft = this.Graph.Extract(this.Derivate, lengthLeft, ConvNetSharp<T>.Zero);
            var extractRight = this.Graph.Extract(this.Derivate, lengthRight, lengthLeft);

            this.Parents[0].RegisterDerivate(new Reshape<T>(this.Graph, extractLeft, this.Graph.Shape(this.Parents[0])));
            this.Parents[1].RegisterDerivate(new Reshape<T>(this.Graph, extractRight, this.Graph.Shape(this.Parents[1])));
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return base.Evaluate(session);
            }

            this.IsDirty = false;

            var left = this.Parents[0].Evaluate(session);
            var right = this.Parents[1].Evaluate(session);

            var batchSize = Math.Max(left.Shape.Dimensions[3], right.Shape.Dimensions[3]);

            var totalLength = (int)(left.Shape.TotalLength / left.Shape.Dimensions[3] + right.Shape.TotalLength / right.Shape.Dimensions[3]);
            if (this.Result == null || this.lastTotalLength != totalLength)
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, totalLength, batchSize));
            }

            left.Concat(right, this.Result);

            return base.Evaluate(session);
        }

        public override string ToString()
        {
            return $"Concat({this.Parents[0]}, {this.Parents[1]})";
        }
    }
}