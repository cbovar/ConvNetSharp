| **`VS 2017`** | **`VS 2015`** |
|-----------------|---------------------|
|[![Build status](https://ci.appveyor.com/api/projects/status/lcqjebortqnn1wkg?svg=true)](https://ci.appveyor.com/project/cbovar/convnetsharp)|[![Build status](https://ci.appveyor.com/api/projects/status/2vtsgpr9ppo5b4gg?svg=true)](https://ci.appveyor.com/project/cbovar/convnetsharp-0kbf4)|

# ConvNetSharp
Started initially as C# port of [ConvNetJS](https://github.com/karpathy/convnetjs). You can use ConvNetSharp to train and evaluate convolutional neural networks (CNN).

Thank you very much to the original author of ConvNetJS (Andrej Karpathy) and to all the contributors!

Current version requires Visual Studio 2017. You can find a version compatible with Visual Studio 2015 [here](https://github.com/cbovar/ConvNetSharp/tree/vs2015)

## What's new ?

30/05/2017

- Available on [Nuget](https://www.nuget.org/packages/Cognitio.ConvNetSharp.Volume/) in pre-release (i.e. not stable)

20/05/2017

- vs 2017 and vs 2015 solutions are now both on the same branch (using same source code).

27/03/2017

- Volumes have their own project
- Volumes have now 4 dimensions (width, height, channel, **batchSize**)
- Generic on numerics to use single or double precision (`Net<double>` or `Net<float>`)
- GPU implementation. Just add '`GPU`' in the namespace: `using ConvNetSharp.Volume.`**GPU**`.Single;`
- ConvNetSharp.Volume and ConvNetSharp.Core are on .NET Standard
- New way to serialize/deserialize. Basically Net object gives a nested dictionary that can be serialized the way you like.
- ToDo: Implement missing trainers, implement missing layers (e.g. regression)

Tag [v0.2.0](https://github.com/cbovar/ConvNetSharp/tree/v0.2.0) was created just before commiting new version.

## Example Code

Here's a minimum example of defining a **2-layer neural network** and training
it on a single data point:
```c#
  // species a 2-layer neural network with one hidden layer of 20 neurons
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
  var x = new Volume(new[] { 0.3, -0.5 }, new Shape(2));
  
  var prob = net.Forward(x);
  
  // prob is a Volume. Volumes have a property Weights that stores the raw data, and WeightGradients that stores gradients
  Console.WriteLine("probability that x is class 0: " + prob.Get(0)); // prints e.g. 0.50101
  
  var trainer = new SgdTrainer(net) { LearningRate = 0.01, L2Decay = 0.001 };
  trainer.Train(x, new Volume(new[] { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, new Shape(1, 1, 10, 1))); // train the network, specifying that x is class zero
  
  var prob2 = net.Forward(x);
  Console.WriteLine("probability that x is class 0: " + prob2.Get(0));
  // now prints 0.50374, slightly higher than previous 0.50101: the networks
  // weights have been adjusted by the Trainer to give a higher probability to
  // the class we trained the network with (zero)
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

Switch to GPU mode simply by adding '`GPU`' in the namespace: `using ConvNetSharp.Volume.`**GPU**`.Single;` or `using ConvNetSharp.Volume.`**GPU**`.Double;`

You must have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.

Mnist GPU demo [here](https://github.com/cbovar/ConvNetSharp/tree/master/Examples/MnistDemo.GPU)

## Save and Load Network
### JSON serialization (not supported by FluentNet)
```c#
// Serialize to json 
var json = net.ToJsonN();

// Deserialize from json
Net deserialized = SerializationExtensions.FromJson<double>(json);
```
