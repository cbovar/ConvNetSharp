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
#if DEBUG
            var inputs = input.ToArray();
            foreach (var i in inputs)
                if (Ops<T>.IsInvalid(i))
                    throw new ArgumentException("Invalid input!");
#endif

            this.InputActivation = input;

            var outputShape = new Shape(this.OutputWidth, this.OutputHeight, this.OutputDepth, input.Shape.Dimensions[3]);

            if (this.OutputActivation == null ||
                !this.OutputActivation.Shape.Equals(outputShape))
            {
                this.OutputActivation = BuilderInstance<T>.Volume.SameAs(input.Storage, outputShape);
            }

            if (this.InputActivationGradients == null ||
                !this.InputActivationGradients.Shape.Equals(input.Shape))
            {
                this.InputActivationGradients = BuilderInstance<T>.Volume.SameAs(this.InputActivation.Storage,
                    this.InputActivation.Shape);
            }

            this.OutputActivation = this.Forward(input, isTraining);

            return this.OutputActivation;
        }

        protected abstract Volume<T> Forward(Volume<T> input, bool isTraining = false);

        public virtual Volume<T> Forward(bool isTraining)
        {
            return this.DoForward(this.Parents[0].Forward(isTraining), isTraining);
        }

        public static LayerBase<T> FromData(IDictionary<string, object> dico)
        {
            var typeName = dico["Type"] as string;
            var type = Type.GetType(typeName);
            var t = (LayerBase<T>)Activator.CreateInstance(type, dico);
            return t;
        }

        public virtual Dictionary<string, object> GetData()
        {
            var dico = new Dictionary<string, object>
            {
                ["Type"] = this.GetType().FullName,
                ["InputHeight"] = this.InputHeight,
                ["InputWidth"] = this.InputWidth,
                ["InputDepth"] = this.InputDepth,
                ["OutputWidth"] = this.OutputWidth,
                ["OutputHeight"] = this.OutputHeight,
                ["OutputDepth"] = this.OutputDepth
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