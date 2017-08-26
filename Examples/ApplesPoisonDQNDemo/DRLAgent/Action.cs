using System;

namespace DeepQLearning.DRLAgent
{
    [Serializable]
    public struct Action
    {
        public int action;
        public double value;
    };
}
