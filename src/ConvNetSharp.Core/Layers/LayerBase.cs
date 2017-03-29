using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public abstract class LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        protected bool IsInitialized;

        protected LayerBase()
        {
        }

        protected LayerBase(Dictionary<string, object> data)
        {
            this.InputWidth = Convert.ToInt32(data["InputWidth"]);
            this.InputHeight = Convert.ToInt32(data["InputHeight"]);
            this.InputDepth = Convert.ToInt32(data["InputDepth"]);
            this.OutputWidth = Convert.ToInt32(data["OutputWidth"]);
            this.OutputDepth = Convert.ToInt32(data["OutputDepth"]);
            this.OutputHeight = Convert.ToInt32(data["OutputHeight"]);
        }

        public Volume<T> InputActivation { get; protected set; }

        public Volume<T> InputActivationGradients { get; protected set; }

        public Volume<T> OutputActivation { get; protected set; }

        public Volume<T> OutputActivationGradients { get; protected set; }

        public int OutputDepth { get; protected set; }

        public int OutputWidth { get; protected set; }

        public int OutputHeight { get; protected set; }

        public int InputDepth { get; private set; }

        public int InputWidth { get; private set; }

        public int InputHeight { get; private set; }

        public LayerBase<T> Child { get; set; }

        public List<LayerBase<T>> Parents { get; set; } = new List<LayerBase<T>>();

        public abstract void Backward(Volume<T> outputGradient);

        internal void ConnectTo(LayerBase<T> layer)
        {
            this.Child = layer;
            layer.Parents.Add(this);

            layer.Init(this.OutputWidth, this.OutputHeight, this.OutputDepth);
        }

        public virtual Volume<T> DoForward(Volume<T> input, bool isTraining = false)
        {
            this.InputActivation = input;

            if (this.OutputActivation == null)
            {
                var shape = new Shape(input.Shape);
                if (shape.DimensionCount > 0)
                    shape.SetDimension(0, this.OutputWidth);
                if (shape.DimensionCount > 1)
                    shape.SetDimension(1, this.OutputHeight);
                if (shape.DimensionCount > 2)
                    shape.SetDimension(2, this.OutputDepth);

                this.OutputActivation = BuilderInstance<T>.Volume.SameAs(input.Storage, shape);
            }

            if (this.InputActivationGradients == null || !Equals(this.InputActivationGradients.Shape, input.Shape))
            {
                this.InputActivationGradients = BuilderInstance<T>.Volume.SameAs(this.InputActivation.Storage, this.InputActivation.Shape);
            }

            this.OutputActivation = Forward(input, isTraining);

            return this.OutputActivation;
        }

        protected abstract Volume<T> Forward(Volume<T> input, bool isTraining = false);

        public virtual Volume<T> Forward(bool isTraining)
        {
            return DoForward(this.Parents[0].Forward(isTraining), isTraining);
        }

        public static LayerBase<T> FromData(IDictionary<string, object> dico)
        {
            var typeName = dico["Type"] as string;
            var type = Type.GetType(typeName);
            var t = Activator.CreateInstance(type, dico) as LayerBase<T>;

            return t;
        }

        public virtual Dictionary<string, object> GetData()
        {
            var dico = new Dictionary<string, object>
            {
                ["Type"] = GetType().FullName,
                ["InputHeight"] = this.InputHeight,
                ["InputWidth"] = this.InputWidth,
                ["InputDepth"] = this.InputDepth,
                ["OutputWidth"] = this.OutputWidth,
                ["OutputHeight"] = this.OutputHeight,
                ["OutputDepth"] = this.OutputDepth,
            };
            return dico;
        }

        public virtual List<ParametersAndGradients<T>> GetParametersAndGradients()
        {
            return new List<ParametersAndGradients<T>>();
        }

        public virtual void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            this.InputWidth = inputWidth;
            this.InputHeight = inputHeight;
            this.InputDepth = inputDepth;
            this.IsInitialized = true;
        }
    }
}