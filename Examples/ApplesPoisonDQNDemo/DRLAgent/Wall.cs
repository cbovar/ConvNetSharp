using System;

namespace DeepQLearning.DRLAgent
{
    // Wall is made up of two points
    [Serializable]
    public class Wall
    {
        public Vector point1, point2;

        public Wall(Vector p1, Vector p2)
        {
            point1 = p1;
            point2 = p2;
        }
    }
}
