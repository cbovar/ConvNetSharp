using System;
using System.Collections.Generic;

namespace ConvnetSharpOLD
{
    [Serializable]
    public class TrainingOptions
    {
        public int temporalWindow = int.MinValue;
        public int experienceSize = int.MinValue;
        public int startLearnThreshold = int.MinValue;
        public int learningStepsTotal = int.MinValue;
        public int learningStepsBurnin = int.MinValue;
        public int[] hiddenLayerSizes;

        public double gamma = double.MinValue;
        public double learningRate = double.MinValue;
        public double epsilonMin = double.MinValue;
        public double epsilonTestTime = double.MinValue;

        public Options options;
        public List<LayerDefinition> layerDefinitions;
        public List<double> radmomActionDistribution;
    }
}
