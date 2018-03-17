using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    public class Dropout<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private Shape _lastInputShape;

        public Dropout(Op<T> x, T dropoutProbability)
        {
            AddParent(x);
            this.DropoutProbability = dropoutProbability;
        }

        public override string Representation => $"Dropout({this.DropoutProbability})";

        public T DropoutProbability { get; set; }

        public override void Differentiate()
        {
            this.Parents[0].RegisterDerivate(new DropoutGradient<T>(this, this.Derivate));
        }

        public Dropout(Dictionary<string, object> data)
        {
            this.DropoutProbability = (T)Convert.ChangeType(data["DropoutProbability"], typeof(T));
        }

        public override Dictionary<string, object> GetData()
        {
            var data = base.GetData();
            data["DropoutProbability"] = this.DropoutProbability;
            return data;
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return this.Result;
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

            return this.Result;
        }
    }
}