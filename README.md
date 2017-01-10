[![Build status](https://ci.appveyor.com/api/projects/status/lcqjebortqnn1wkg?svg=true)](https://ci.appveyor.com/project/cbovar/convnetsharp)

# ConvNetSharp
C# port of [ConvNetJS](https://github.com/karpathy/convnetjs). You can use ConvNetSharp to train and evaluate convolutional neural networks (CNN).

Thank you very much to the original author (Andrej Karpathy) and to all the contributors to ConvNetJS!

## Example Code

Here's a minimum example of defining a **2-layer neural network** and training
it on a single data point:
```c#
  // species a 2-layer neural network with one hidden layer of 20 neurons
  var net = new Net();

  // input layer declares size of input. here: 2-D data
  // ConvNetSharp works on 3-Dimensional volumes (width, height, depth), but if you're not dealing with images
  // then the first two dimensions (width, height) will always be kept at size 1
  net.AddLayer(new InputLayer(1, 1, 2));

  // declare 20 neurons, followed by ReLU (rectified linear unit non-linearity)
  net.AddLayer(new FullyConnLayer(20, Activation.Relu));

  // declare the linear classifier on top of the previous hidden layer
  net.AddLayer(new SoftmaxLayer(10));

  // forward a random data point through the network
  var x = new Volume(new[] {0.3, -0.5});

  var prob = net.Forward(x);

  // prob is a Volume. Volumes have a property Weights that stores the raw data, and WeightGradients that stores gradients
  Console.WriteLine("probability that x is class 0: " + prob.Weights[0]); // prints e.g. 0.50101

  var trainer = new SgdTrainer(net) {LearningRate = 0.01, L2Decay = 0.001};
  trainer.Train(x, 0); // train the network, specifying that x is class zero

  var prob2 = net.Forward(x);
  Console.WriteLine("probability that x is class 0: " + prob2.Weights[0]);
  // now prints 0.50374, slightly higher than previous 0.50101: the networks
  // weights have been adjusted by the Trainer to give a higher probability to
  // the class we trained the network with (zero)
```
