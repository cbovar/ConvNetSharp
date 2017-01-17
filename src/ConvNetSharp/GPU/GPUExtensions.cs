using ConvNetSharp.Fluent;
using ConvNetSharp.GPU.Layers;
using ConvNetSharp.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNetSharp.GPU
{
    public static class GPUExtensions
    {
        public static Net ToGPU(this Net net)
        {
            var layers = net.Layers;

            for (int i = 0; i < layers.Count; i++)
            {
                var convlayer = layers[i] as ConvLayer;
                if (convlayer != null)
                {
                    layers.RemoveAt(i);
                    var gpu = new ConvLayerGPU(convlayer);
                    layers.Insert(i, gpu);
                    gpu.Init(layers[i - 1].OutputWidth, layers[i - 1].OutputHeight, layers[i - 1].OutputDepth);
                }
            }

            return net;
        }

        public static Net ToNonGPU(this Net net)
        {
            var layers = net.Layers;

            for (int i = 0; i < layers.Count; i++)
            {
                var convlayer = layers[i] as ConvLayerGPU;
                if (convlayer != null)
                {
                    layers.RemoveAt(i);
                    var nongpu = new ConvLayer(convlayer.Width, convlayer.Height, convlayer.FilterCount);
                    nongpu.Filters = convlayer.Filters;
                    nongpu.Biases = convlayer.Biases;
                    layers.Insert(i, nongpu);
                    nongpu.Init(layers[i - 1].OutputWidth, layers[i - 1].OutputHeight, layers[i - 1].OutputDepth);
                }
            }

            return net;
        }

        public static FluentNet ToGPU(this FluentNet net)
        {
            var layers = net.Layers;

            for (int i = 0; i < layers.Count; i++)
            {
                var convlayer = layers[i] as ConvLayer;
                if (convlayer != null)
                {
                    var gpu = new ConvLayerGPU(convlayer);
                    net.ReplaceLayer(convlayer, gpu);
                }
            }

            return net;
        }

        public static FluentNet ToNonGPU(this FluentNet net)
        {
            var layers = net.Layers;

            for (int i = 0; i < layers.Count; i++)
            {
                var convlayer = layers[i] as ConvLayerGPU;
                if (convlayer != null)
                {
                    var nongpu = new ConvLayer(convlayer.Width, convlayer.Height, convlayer.FilterCount);
                    nongpu.Filters = convlayer.Filters;
                    nongpu.Biases = convlayer.Biases;

                    net.ReplaceLayer(convlayer, nongpu);
                }
            }

            return net;
        }
    }
}
