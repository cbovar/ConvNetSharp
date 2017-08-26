namespace ConvNetSharp.ReinforcementLearning.Deep
{
    /// <summary>
    /// Representation of a policy in the context of the action-value-function.
    /// </summary>
    public struct ActionValuePolicy
    {
        public int Action
        {
            get; set;
        }

        /// <summary>
        /// Expected future reward for following the given action.
        /// </summary>
        public double Value
        {
            get; set;
        }
    }
}