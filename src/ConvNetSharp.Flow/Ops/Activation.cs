using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    /// <summary>
    ///     y = f(x) where f can be Sigmoid ,Relu,Tanh or ClippedRelu
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Activation<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Activation(ConvNetSharp<T> graph, Dictionary<string, object> data) : base(graph)
        {
            Enum.TryParse((string)data["ActivationType"], out ActivationType result);

            this.Type = result;
        }

        public Activation(ConvNetSharp<T> graph, Op<T> x, ActivationType type) : base(graph)
        {
            this.AddParent(x);
            this.Type = type;
        }

        public ActivationType Type { get; set; }

        public override string Representation => $"{this.Type}";

        public override void Differentiate()
        {
            this.Parents[0].RegisterDerivate(new ActivationGradient<T>(this.Graph, this.Parents[0], this, this.Derivate, this.Type));
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                this.Result?.Dispose();
            }

            base.Dispose(disposing);
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            if (!this.IsDirty)
            {
                return base.Evaluate(session);
            }

            this.IsDirty = false;

            var x = this.Parents[0].Evaluate(session);

            if (this.Result == null || !Equals(this.Result.Shape, x.Shape))
            {
                this.Result?.Dispose();
                this.Result = BuilderInstance<T>.Volume.SameAs(x.Shape);
            }

            x.Activation(this.Type, this.Result);
            return base.Evaluate(session);
        }

        public override Dictionary<string, object> GetData()
        {
            var data = base.GetData();
            data["ActivationType"] = this.Type;
            return data;
        }

        public override string ToString()
        {
            return $"{this.Type}({this.Parents[0]})";
        }
    }
}