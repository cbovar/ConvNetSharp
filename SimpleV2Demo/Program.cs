using ConvNetSharp.V2;

internal class Program
{
    private static void Main()
    {
        // Initialize ConvNetSharp with default settings for float type
        var convNet = new ConvNetSharp<float>();

        // Set active ConvNetSharp instance for the scope
        using (ConvNetSharp<float>.SetActive(convNet))
        {
            // Create a session
            using var session = new Session<float>();

            // Build the computation graph
            var input = convNet.PlaceHolder("input"); // Placeholder for input data
            var target = convNet.PlaceHolder("target"); // Placeholder for target values

            // Variables for weights and bias
            var weights = convNet.Variable(new Shape(3, 1), "weights", true);
            var bias = convNet.Variable(new Shape(1, 1), "bias", true);

            // Linear model: y = x * weights + bias
            var predicted = convNet.MatMult(input, convNet.Transpose(weights)) + bias;

            // Loss function: Mean Squared Error
            var loss = convNet.Sum(convNet.Power(predicted - target, 2), new Shape(1, 1));

            // Initialize placeholders with mock data
            var inputData = Volume<float>.Create(new[] { 1.0f, 2.0f, 3.0f }, new Shape(3, 1));
            var targetData = Volume<float>.Create(new[] { 2.0f }, new Shape(1, 1));
            session.InitializePlaceHolders(loss, new Dictionary<string, Volume<float>>
            {
                { "input", inputData },
                { "target", targetData }
            });

            // Perform a forward pass
            var forwardResult = session.Run(loss, null);
            Console.WriteLine($"Initial loss: {forwardResult.Get(0)}");

            // Perform differentiation to compute gradients
            session.Differentiate(loss);


            // Manually update weights and bias using simple gradient descent
            const float learningRate = 0.01f;
            foreach (var variable in session.LearnableVariables.Values)
            {
                // Gradient descent: variable = variable - learningRate * gradient
                var gradient = variable.Derivative.Evaluate(session);
                variable.Result = variable.Result - gradient * learningRate;
            }

            // Run again after an update
            var updatedResult = session.Run(loss, null);
            Console.WriteLine($"Loss after update: {updatedResult.Get(0)}");
        }
    }
}