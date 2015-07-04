using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;

namespace ConvNetSharp.Tests
{
    public static class GradientCheckTools
    {
        public static void GradientCheck(LayerBase layer, int inputWidth, int inputHeight, int inputDepth, double epsilon = 1e-4)
        {
            layer.Init(inputWidth, inputHeight, inputDepth);

            // Forward pass
            var input = new Volume(inputWidth, inputHeight, inputDepth);
            var output = layer.Forward(input, true);

            // Set output gradients to 1
            for (var n = 0; n < output.WeightGradients.Length; n++)
            {
                output.WeightGradients[n] = 1.0;
            }

            // Backward pass to retrieve gradients
            layer.Backward();
            var computedGradients = input;

            // Now let's approximate gradient using derivate definition
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
                        var actual = computedGradients.GetGradient(x, y, d);
                        Assert.AreEqual(gradient, actual, 1e-4); // compare layer gradient to the approximated gradient
                    }
                }
            }
        }

        public static void GradienWrtParameterstCheck(int inputWidth, int inputHeight, int inputDepth, LayerBase layer, double epsilon = 1e-4)
        {
            layer.Init(inputWidth, inputHeight, inputDepth);

            // Forward pass
            var input = new Volume(inputWidth, inputHeight, inputDepth, 1.0);
            var output = layer.Forward(input);

            // Set output gradients to 1
            for (var n = 0; n < output.WeightGradients.Length; n++)
            {
                output.WeightGradients[n] = 1.0;
            }

            // Backward pass to retrieve gradients
            layer.Backward();

            List<ParametersAndGradients> paramsAndGrads = layer.GetParametersAndGradients();

            foreach (var paramAndGrad in paramsAndGrads)
            {
                double[] computedGradients = paramAndGrad.Gradients;

                // Now let's approximate gradient
                for (var i = 0; i < paramAndGrad.Parameters.Length; i++)
                {
                    input = new Volume(inputWidth, inputHeight, inputDepth, 1.0);

                    var oldValue = paramAndGrad.Parameters[i];
                    paramAndGrad.Parameters[i] = oldValue + epsilon;
                    var output1 = layer.Forward(input);
                    paramAndGrad.Parameters[i] = oldValue - epsilon;
                    var output2 = layer.Forward(input);
                    paramAndGrad.Parameters[i] = oldValue;

                    output1.AddFromScaled(output2, -1.0); // output1 = output1 - output2

                    var grad = new double[output.WeightGradients.Length];
                    for (var j = 0; j < output.WeightGradients.Length; j++)
                    {
                        grad[j] = output1.Weights[j] / (2.0 * epsilon);
                    }

                    var gradient = grad.Sum(); // approximated gradient
                    Assert.AreEqual(gradient, computedGradients[i], 1e-4); // compare layer gradient to the approximated gradient
                }
            }
        }
    }
}