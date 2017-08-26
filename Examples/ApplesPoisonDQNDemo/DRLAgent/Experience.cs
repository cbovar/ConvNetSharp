using System;

namespace DeepQLearning.DRLAgent
{
    // An agent is in state0 and does action0
    // environment then assigns reward0 and provides new state, state1
    // Experience nodes store all this information, which is used in the
    // Q-learning update step

    /// <summary>
    /// An experience stores the initial state of an agent, the chosen action, the received received action and the final state of the agent.
    /// </summary>
    [Serializable]
    public class Experience
    {
        public double[] initialState;
        public int initialAction;
        public double initialReward;
        public double[] finalState;

        public Experience()
        {

        }

        public Experience(double[] state0, int action0, double reward0, double[] state1)
        {
            initialState = state0;
            initialAction = action0;
            initialReward = reward0;
            finalState = state1;
        }
    }

}
