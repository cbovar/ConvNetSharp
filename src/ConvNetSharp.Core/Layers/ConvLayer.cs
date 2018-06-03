using System;
using System.Collections.Generic;
using ConvNetSharp.Core.Serialization;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public class ConvLayer<T> : LayerBase<T>, IDotProductLayer<T> where T : struct, IEquatable<T>, IFormattable
    {
        private T _biasPref;
        private int _pad;
        private int _stride = 1;

        public ConvLayer(Dictionary<string, object> data) : base(data)
        {
            this.L1DecayMul = Ops<T>.Zero;
            this.L2DecayMul = Ops<T>.One;

            this.FilterCount = Convert.ToInt32(data["FilterCount"]);
            this.Width = Convert.ToInt32(data["Width"]);
            this.Height = Convert.ToInt32(data["Height"]);
            this.Stride = Convert.ToInt32(data["Stride"]);
            this.Pad = Convert.ToInt32(data["Pad"]);
            this.Filters = BuilderInstance<T>.Volume.From(data["Filters"].ToArrayOfT<T>(), new Shape(this.Width, this.Height, this.InputDepth, this.FilterCount));
            this.Bias = BuilderInstance<T>.Volume.From(data["Bias"].ToArrayOfT<T>(), new Shape(1, 1, this.FilterCount));
            this.BiasPref = (T) Convert.ChangeType(data["BiasPref"], typeof(T));
            this.FiltersGradient = BuilderInstance<T>.Volume.From(data["FiltersGradient"].ToArrayOfT<T>(), new Shape(this.Width, this.Height, this.InputDepth, this.FilterCount));
            this.BiasGradient = BuilderInstance<T>.Volume.From(data["BiasGradient"].ToArrayOfT<T>(), new Shape(1, 1, this.FilterCount));

            this.IsInitialized = true;
        }

        public ConvLayer(int width, int height, int filterCount)
        {
            this.L1DecayMul = Ops<T>.Zero;
            this.L2DecayMul = Ops<T>.One;

            this.FilterCount = filterCount;
            this.Width = width;
            this.Height = height;
        }

        public int Width { get; }

        public int Height { get; }

        public Volume<T> Bias { get; private set; }

        public Volume<T> BiasGradient { get; private set; }

        public Volume<T> Filters { get; private set; }

        public Volume<T> FiltersGradient { get; private set; }

        public int FilterCount { get; }

        public T L1DecayMul { get; set; }

        public T L2DecayMul { get; set; }

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

        public T BiasPref
        {
            get { return this._biasPref; }
            set
            {
                this._biasPref = value;

                if (this.IsInitialized)
                {
                    UpdateOutputSize();
                }
            }
        }

        public override void Backward(Volume<T> outputGradient)
        {
            this.OutputActivationGradients = outputGradient;

            // compute gradient wrt weights and data
            this.InputActivation.ConvolutionGradient(this.Filters, this.OutputActivationGradients,
                this.FiltersGradient, this.Pad, this.Stride, this.InputActivationGradients);
            this.OutputActivationGradients.BiasGradient(this.BiasGradient);
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            input.Convolution(this.Filters, this.Pad, this.Stride, this.OutputActivation);
            this.OutputActivation.Add(this.Bias, this.OutputActivation);
            return this.OutputActivation;
        }

        public override Dictionary<string, object> GetData()
        {
            var dico = base.GetData();

            dico["Width"] = this.Width;
            dico["Height"] = this.Height;
            dico["FilterCount"] = this.FilterCount;
            dico["Stride"] = this.Stride;
            dico["Pad"] = this.Pad;
            dico["Bias"] = this.Bias.ToArray();
            dico["Filters"] = this.Filters.ToArray();
            dico["BiasPref"] = this.BiasPref;
            dico["FiltersGradient"] = this.FiltersGradient.ToArray();
            dico["BiasGradient"] = this.BiasGradient.ToArray();

            return dico;
        }

        public override List<ParametersAndGradients<T>> GetParametersAndGradients()
        {
            var response = new List<ParametersAndGradients<T>>
            {
                new ParametersAndGradients<T>
                {
                    Volume = this.Filters,
                    Gradient = this.FiltersGradient,
                },
                new ParametersAndGradients<T>
                {
                    Volume = this.Bias,
                    Gradient = this.BiasGradient,
                }
            };

            return response;
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            UpdateOutputSize();
        }

        internal void UpdateOutputSize()
        {
            // required
            this.OutputDepth = this.FilterCount;

            // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
            // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
            // final application.
            this.OutputWidth =
                (int) Math.Floor((this.InputWidth + this.Pad * 2 - this.Width) / (double) this.Stride + 1);
            this.OutputHeight =
                (int) Math.Floor((this.InputHeight + this.Pad * 2 - this.Height) / (double) this.Stride + 1);

            // initializations
            var scale = Math.Sqrt(2.0 / (this.Width * this.Height * this.InputDepth));

            var shape = new Shape(this.Width, this.Height, this.InputDepth, this.OutputDepth);
            if (this.Filters == null || !this.Filters.Shape.Equals(shape))
            {
                this.Filters?.Dispose();
                this.Filters = BuilderInstance<T>.Volume.Random(shape, 0.0, scale);
                this.FiltersGradient?.Dispose();
                this.FiltersGradient = BuilderInstance<T>.Volume.SameAs(shape);
            }

            this.Bias = BuilderInstance<T>.Volume.From(new T[this.OutputDepth].Populate(this.BiasPref),
                new Shape(1, 1, this.OutputDepth));
            this.BiasGradient = BuilderInstance<T>.Volume.SameAs(new Shape(1, 1, this.OutputDepth));
        }
    }
}