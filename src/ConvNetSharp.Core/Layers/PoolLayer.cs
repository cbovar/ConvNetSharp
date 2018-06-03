using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public class PoolLayer<T> : LayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        private int _pad;
        private int _stride = 2;

        public PoolLayer(Dictionary<string, object> data) : base(data)
        {
            this.Width =  Convert.ToInt32(data["Width"]);
            this.Height =  Convert.ToInt32(data["Height"]);
            this.Pad =  Convert.ToInt32(data["Pad"]);
            this.Stride =  Convert.ToInt32(data["Stride"]);
            this.IsInitialized = true;
        }

        public PoolLayer(int width, int height)
        {
            this.Width = width;
            this.Height = height;
        }

        public int Width { get; }

        public int Height { get; }

        public int Stride
        {
            get { return this._stride; }
            set
            {
                this._stride = value;
                if (this.IsInitialized)
                {
                    UpdateOutputSize();
                }
            }
        }

        public int Pad
        {
            get { return this._pad; }
            set
            {
                this._pad = value;
                if (this.IsInitialized)
                {
                    UpdateOutputSize();
                }
            }
        }

        public override void Backward(Volume<T> outputGradient)
        {
            this.OutputActivationGradients = outputGradient;

            this.InputActivationGradients.Clear();

            this.OutputActivation.PoolGradient(this.InputActivation, this.OutputActivationGradients, this.Width,
                this.Height, this.Pad, this.Pad, this.Stride, this.Stride, this.InputActivationGradients);
        }

        public override Dictionary<string, object> GetData()
        {
            var dico = base.GetData();

            dico["Width"] = this.Width;
            dico["Height"] = this.Height;
            dico["Stride"] = this.Stride;
            dico["Pad"] = this.Pad;

            return dico;
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            input.Pool(this.Width, this.Height, this.Pad, this.Pad, this.Stride, this.Stride, this.OutputActivation);
            return this.OutputActivation;
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            UpdateOutputSize();
        }

        private void UpdateOutputSize()
        {
            // computed
            this.OutputDepth = this.InputDepth;
            this.OutputWidth = (int)Math.Floor((this.InputWidth + this.Pad * 2 - this.Width) / (double)this.Stride + 1);
            this.OutputHeight = (int)Math.Floor((this.InputHeight + this.Pad * 2 - this.Height) / (double)this.Stride + 1);
        }
    }
}