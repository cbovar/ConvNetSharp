using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Dropout<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private Shape _lastInputShape;

        public Dropout(ConvNetSharp<T> graph, Op<T> x, T dropoutProbability) : base(graph)
        {
            AddParent(x);
            this.DropoutProbability = dropoutProbability;
        }

        public Dropout(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
            this.DropoutProbability = (T) Convert.ChangeType(data["DropoutProbability"], typeof(T));
        }

        public override string Representation => $"Dropout({this.DropoutProbability})";

        public T DropoutProbability { get; set; }

        public override void Differentiate()
        {
            this.Parents[0].RegisterDerivate(new DropoutGradient<T>(this.Graph, this, this.Derivate));
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return base.Evaluate(session);
            }

            this.IsDirty = false;

            var x = this.Parents[0].Evaluate(session);

            if (this.Result == null || !Equals(this._lastInputShape, x.Shape))
            {
                this._lastInputShape = new Shape(x.Shape);

                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(x.Shape);
            }

            x.DoDropout(this.Result, session.IsTraining, this.DropoutProbability);

            return base.Evaluate(session);
        }

        public override Dictionary<string, object> GetData()
        {
            var data = base.GetData();
            data["DropoutProbability"] = this.DropoutProbability;
            return data;
        }

        public override string ToString()
        {
            return $"Dropout({this.Parents[0]})";
        }
    }
}