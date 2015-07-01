using System.Linq;
using NUnit.Framework;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class FullyConnLayerTests
    {
        [Test]
        public void GradientWrtInputCheck()
        {
            const int inputWidth = 2;
            const int inputHeight = 2;
            const int inputDepth = 2;
            const int inputCount = inputWidth * inputHeight * inputDepth;
            const int neuronCount = 2;

            // Create layer
            var layer = new FullyConnLayer(neuronCount);
            layer.Init(inputWidth, inputHeight, inputDepth);

            // Forward pass
            var input = new Volume(inputWidth, inputHeight, inputDepth, 1.0);
            var output = layer.Forward(input);

            // Set output gradients to 1
            for (var n = 0; n < neuronCount; n++)
            {
                output.WeightGradients[n] = 1.0;
            }

            // Backward pass to retrieve gradients
            layer.Backward();
            double[] computedGradients = input.WeightGradients;

            // Now let's approximate gradient using derivate definition
            const double epsilon = 1e-4;
            for (var i = 0; i < inputCount; i++)
            {
                var input1 = new Volume(inputWidth, inputHeight, inputDepth, 1.0);
                var input2 = new Volume(inputWidth, inputHeight, inputDepth, 1.0);

                input1.Weights[i] = 1.0 + epsilon;
                input2.Weights[i] = 1.0 - epsilon;

                var output1 = layer.Forward(input1);
                var output2 = layer.Forward(input2);
                output1.AddFromScaled(output2, -1.0); // output1 = output1 - output2

                var grad = new double[neuronCount];
                for (var j = 0; j < neuronCount; j++)
                {
                    grad[j] = output1.Weights[j] / (2.0 * epsilon);
                }

                var gradient = grad.Sum(); // approximated gradient
                Assert.AreEqual(gradient, computedGradients[i], 1e-4); // compare layer gradient to the approximated gradient
            }
        }

        [Test]
        public void GradientWrtFiltersCheck()
        {
            const int inputWidth = 2;
            const int inputHeight = 2;
            const int inputDepth = 2;
            const int inputCount = inputWidth * inputHeight * inputDepth;
            const int neuronCount = 2;

            // Create layer
            var layer = new FullyConnLayer(neuronCount);
            layer.Init(inputWidth, inputHeight, inputDepth);

            // Forward pass
            var input = new Volume(inputWidth, inputHeight, inputDepth, 1.0);
            var output = layer.Forward(input);

            // Set output gradients to 1
            for (var n = 0; n < neuronCount; n++)
            {
                output.WeightGradients[n] = 1.0;
            }

            // Backward pass to retrieve gradients
            layer.Backward();

            for (var n = 0; n < neuronCount; n++)
            {
                double[] computedGradients = layer.Filters[n].WeightGradients;

                // Now let's approximate gradient
                const double epsilon = 1e-4;
                for (var i = 0; i < inputCount; i++)
                {
                    input = new Volume(inputWidth, inputHeight, inputDepth, 1.0);

                    var oldValue = layer.Filters[n].Weights[i];
                    layer.Filters[n].Weights[i] = oldValue + epsilon;
                    var output1 = layer.Forward(input);
                    layer.Filters[n].Weights[i] = oldValue - epsilon;
                    var output2 = layer.Forward(input);
                    layer.Filters[n].Weights[i] = oldValue;

                    output1.AddFromScaled(output2, -1.0); // output1 = output1 - output2

                    var grad = new double[neuronCount];
                    for (var j = 0; j < neuronCount; j++)
                    {
                        grad[j] = output1.Weights[j] / (2.0 * epsilon);
                    }

                    var gradient = grad.Sum(); // approximated gradient
                    Assert.AreEqual(gradient, computedGradients[i], 1e-4); // compare layer gradient to the approximated gradient
                }
            }
        }

        [Test]
        public void GradientWrtBiasesCheck()
        {
            const int inputWidth = 2;
            const int inputHeight = 2;
            const int inputDepth = 2;
            const int neuronCount = 2;

            // Create layer
            var layer = new FullyConnLayer(neuronCount);
            layer.Init(inputWidth, inputHeight, inputDepth);

            // Forward pass
            var input = new Volume(inputWidth, inputHeight, inputDepth, 1.0);
            var output = layer.Forward(input);

            // Set output gradients to 1
            for (var n = 0; n < neuronCount; n++)
            {
                output.WeightGradients[n] = 1.0;
            }

            // Backward pass to retrieve gradients
            layer.Backward();

            double[] computedGradients = layer.Biases.WeightGradients;

            // Now let's approximate gradient
            const double epsilon = 1e-4;
            for (var i = 0; i < neuronCount; i++)
            {
                input = new Volume(inputWidth, inputHeight, inputDepth, 1.0);

                var oldValue = layer.Biases.Weights[i];
                layer.Biases.Weights[i] = oldValue + epsilon;
                var output1 = layer.Forward(input);
                layer.Biases.Weights[i] = oldValue - epsilon;
                var output2 = layer.Forward(input);
                layer.Biases.Weights[i] = oldValue;

                output1.AddFromScaled(output2, -1.0); // output1 = output1 - output2

                var grad = new double[neuronCount];
                for (var j = 0; j < neuronCount; j++)
                {
                    grad[j] = output1.Weights[j] / (2.0 * epsilon);
                }

                var gradient = grad.Sum(); // approximated gradient
                Assert.AreEqual(gradient, computedGradients[i], 1e-4); // compare layer gradient to the approximated gradient
            }
        }
    }
}