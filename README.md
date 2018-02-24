<div align="center">
  <img src="https://github.com/cbovar/ConvNetSharp/blob/master/img/ConvNetSharpLogo.png"><br><br>
</div>

-----------------

| **`VS 2017`** | **`VS 2015`** | **`Nuget`** |
|-----------------|---------------------|------------------|
|[![Build status](https://ci.appveyor.com/api/projects/status/lcqjebortqnn1wkg?svg=true)](https://ci.appveyor.com/project/cbovar/convnetsharp)|[![Build status](https://ci.appveyor.com/api/projects/status/2vtsgpr9ppo5b4gg?svg=true)](https://ci.appveyor.com/project/cbovar/convnetsharp-0kbf4)|[![ConvNetSharp NuGet package](https://img.shields.io/nuget/v/Cognitio.ConvNetSharp.Core.svg?style=flat)](https://www.nuget.org/packages/Cognitio.ConvNetSharp.Core/) [![ConvNetSharp NuGet package](https://img.shields.io/nuget/dt/Cognitio.ConvNetSharp.Core.svg?style=flat)](https://www.nuget.org/packages/Cognitio.ConvNetSharp.Core/)|


# ConvNetSharp
Started initially as C# port of [ConvNetJS](https://github.com/karpathy/convnetjs). You can use ConvNetSharp to train and evaluate convolutional neural networks (CNN).

Thank you very much to the original author of ConvNetJS (Andrej Karpathy) and to all the contributors!

&#x1F539; NEW &#x1F539; See **ConvNetSharp** in action driving a car ! [Micromachine.AI](https://github.com/cbovar/Micromachine.AI)

### 3 ways to create neural network

| Core.Layers  | Flow.Layers | Computation graph |
| ------------- | ------------- | ------------- |
| No computation graph  | Layers that create a computation graph behind the scene  | 'Pure flow'  |
| Network organised by stacking layers  | Network organised by stacking layers  | 'Ops' connected to each others. Can implement more complex networks  |
| ![Layers](https://github.com/cbovar/ConvNetSharp/blob/master/img/structure.png) |  ![Layers](https://github.com/cbovar/ConvNetSharp/blob/master/img/structure.png)  | ![Layers](https://github.com/cbovar/ConvNetSharp/blob/master/img/graph.png)  |
| E.g. [MnistDemo](https://github.com/cbovar/ConvNetSharp/tree/master/Examples/MnistFlowGPUDemo)  |  E.g. [MnistFlowGPUDemo](https://github.com/cbovar/ConvNetSharp/tree/master/Examples/MnistDemo.Flow.GPU) or [Flow version of Classify2DDemo ](https://github.com/cbovar/ConvNetSharp/blob/master/Examples/FlowDemo/Classify2DDemo.cs)  | E.g. [ExampleCpuSingle](https://github.com/cbovar/ConvNetSharp/blob/master/Examples/FlowDemo/ExampleCPUSingle.cs)  |

## Example Code

Here's a minimum example of defining a **2-layer neural network** and training
it on a single data point:
```c#
using System;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Training.Double;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;

namespace MinimalExample
{
    internal class Program
    {
        private static void Main()
        {
            // specifies a 2-layer neural network with one hidden layer of 20 neurons
            var net = new Net<double>();

            // input layer declares size of input. here: 2-D data
            // ConvNetJS works on 3-Dimensional volumes (width, height, depth), but if you're not dealing with images
            // then the first two dimensions (width, height) will always be kept at size 1
            net.AddLayer(new InputLayer(1, 1, 2));

            // declare 20 neurons
            net.AddLayer(new FullyConnLayer(20));

            // declare a ReLU (rectified linear unit non-linearity)
            net.AddLayer(new ReluLayer());

            // declare a fully connected layer that will be used by the softmax layer
            net.AddLayer(new FullyConnLayer(10));

            // declare the linear classifier on top of the previous hidden layer
            net.AddLayer(new SoftmaxLayer(10));

            // forward a random data point through the network
            var x =  BuilderInstance.Volume.From(new[] { 0.3, -0.5 }, new Shape(2));

            var prob = net.Forward(x);

            // prob is a Volume. Volumes have a property Weights that stores the raw data, and WeightGradients that stores gradients
            Console.WriteLine("probability that x is class 0: " + prob.Get(0)); // prints e.g. 0.50101

            var trainer = new SgdTrainer(net) { LearningRate = 0.01, L2Decay = 0.001 };
            trainer.Train(x, BuilderInstance.Volume.From(new[] { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, new Shape(1, 1, 10, 1))); // train the network, specifying that x is class zero

            var prob2 = net.Forward(x);
            Console.WriteLine("probability that x is class 0: " + prob2.Get(0));
            // now prints 0.50374, slightly higher than previous 0.50101: the networks
            // weights have been adjusted by the Trainer to give a higher probability to
            // the class we trained the network with (zero)
        }
    }
}
```

## Fluent API (see [FluentMnistDemo](https://github.com/cbovar/ConvNetSharp/tree/master/Examples/FluentMnistDemo))

```c#
var net = FluentNet<double>.Create(24, 24, 1)
                   .Conv(5, 5, 8).Stride(1).Pad(2)
                   .Relu()
                   .Pool(2, 2).Stride(2)
                   .Conv(5, 5, 16).Stride(1).Pad(2)
                   .Relu()
                   .Pool(3, 3).Stride(3)
                   .FullyConn(10)
                   .Softmax(10)
                   .Build();
```

## GPU

To switch to GPU mode:
* add '`GPU`' in the namespace: `using ConvNetSharp.Volume.`**GPU**`.Single;` or `using ConvNetSharp.Volume.`**GPU**`.Double;`
* add `BuilderInstance<float>.Volume = new ConvNetSharp.Volume.GPU.Single.VolumeBuilder();` or `BuilderInstance<double>.Volume = new ConvNetSharp.Volume.GPU.Double.VolumeBuilder();` at the beggining of your code

You must have [CUDA version 8](https://developer.nvidia.com/cuda-downloads) and [Cudnn version 6.1](https://developer.nvidia.com/cudnn) installed.
Cudnn bin path should be referenced in the PATH environment variable.

Mnist GPU demo [here](https://github.com/cbovar/ConvNetSharp/tree/master/Examples/MnistDemo.GPU)

## Save and Load Network
### JSON serialization (not supported by FluentNet)
```c#
// Serialize to json 
var json = net.ToJsonN();

// Deserialize from json
Net deserialized = SerializationExtensions.FromJson<double>(json);
```
