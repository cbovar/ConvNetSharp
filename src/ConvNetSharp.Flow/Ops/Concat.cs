﻿using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Concat<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Concat(Dictionary<string, object> data)
        {
        }

        public Concat(Op<T> left, Op<T> right)
        {
            AddParent(left);
            AddParent(right);
        }

        public override void Differentiate()
        {
            var flattenShape = new Shape(1, 1, -1, Shape.Keep);
            var lengthLeft = new Shape<T>(new Reshape<T>(this.Parents[0], flattenShape), 2);
            var lengthRight = new Shape<T>(new Reshape<T>(this.Parents[1], flattenShape), 2);

            var extractLeft = new Extract<T>(this.Derivate, lengthLeft, ConvNetSharp<T>.Zero);
            var extractRight = new Extract<T>(this.Derivate, lengthRight, lengthLeft);

            this.Parents[0].RegisterDerivate(new Reshape<T>(extractLeft, new Shape<T>(this.Parents[0])));
            this.Parents[1].RegisterDerivate(new Reshape<T>(extractRight, new Shape<T>(this.Parents[1])));
        }

        public override string Representation => "Concat";

        private int lastTotalLength = 0;

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return base.Evaluate(session);
            }
            this.IsDirty = false;

            var left = this.Parents[0].Evaluate(session);
            var right = this.Parents[1].Evaluate(session);

            var batchSize = Math.Max(left.Shape.GetDimension(3), right.Shape.GetDimension(3));

            int totalLength = (int)(left.Shape.TotalLength / left.Shape.GetDimension(3) + right.Shape.TotalLength / right.Shape.GetDimension(3));
            if (this.Result == null || this.lastTotalLength != totalLength)
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, totalLength, batchSize));
            }

            left.DoConcat(right, this.Result);

            return base.Evaluate(session);
        }

        public override string ToString()
        {
            return $"Concat({ this.Parents[0]}, { this.Parents[1]})";
        }
    }
}