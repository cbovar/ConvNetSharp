using System.Linq;
using NUnit.Framework;

namespace ConvNetSharp.Tests
{
    [TestFixture]
    public class MaxoutLayerTests
    {
        [Test]
        public void GradientWrtInputCheck()
        {
            const int inputWidth = 20;
            const int inputHeight = 20;
            const int inputDepth = 2;
            const int groupSize = 4;

            // Create layer
            var layer = new MaxoutLayer { GroupSize = groupSize };
            layer.Init(inputWidth, inputHeight, inputDepth);

            // Forward pass
            var input = new Volume(inputWidth, inputHeight, inputDepth);
            var output = layer.Forward(input);

            // Set output gradients to 1
            for (var n = 0; n < output.WeightGradients.Length; n++)
            {
                output.WeightGradients[n] = 1.0;
            }

            // Backward pass to retrieve gradients
            layer.Backward();
            var computedGradients = input;

            // Now let's approximate gradient using derivate definition
            const double epsilon = 1e-6;
            for (var d = 0; d < inputDepth; d++)
            {
                for (var y = 0; y < inputHeight; y++)
                {
                    for (var x = 0; x < inputWidth; x++)
                    {
                        var oldValue = input.Get(x, y, d);

                        input.Set(x, y, d, oldValue + epsilon);
                        var output1 = layer.Forward(input);
                        input.Set(x, y, d, oldValue - epsilon);
                        var output2 = layer.Forward(input);

                        input.Set(x, y, d, oldValue);

                        output1.AddFromScaled(output2, -1.0); // output1 = output1 - output2

                        var grad = new double[output.WeightGradients.Length];
                        for (var j = 0; j < output.WeightGradients.Length; j++)
                        {
                            grad[j] = output1.Weights[j] / (2.0 * epsilon);
                        }

                        var gradient = grad.Sum(); // approximated gradient
                        Assert.AreEqual(gradient, computedGradients.GetGradient(x, y, d), 1e-4); // compare layer gradient to the approximated gradient
                    }
                }
            }
        }
    }
}