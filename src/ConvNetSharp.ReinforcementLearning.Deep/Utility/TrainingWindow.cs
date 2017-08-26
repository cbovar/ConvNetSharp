using System.Collections.Generic;

namespace ConvNetSharp.ReinforcementLearning.Deep.Utility
{
    /// <summary>
    /// A training window stores an amount of values based on a specified size. It is capable of computing average values. For example, it can aid in validating the training accuracy.
    /// </summary>
    public class TrainingWindow
    {
        #region Member Fields
        private List<double> _values;
        private int _size;
        private int _minSize;
        private double _sum = 0;
        #endregion

        #region Constructor
        /// <summary>
        /// Instantiates a TrainingWindow given a certain size.
        /// </summary>
        /// <param name="size">Number of values, which can be stored</param>
        /// <param name="minSize">Minimum number of values to be stored before returning a proper average value.</param>
        public TrainingWindow(int size, int minSize)
        {
            _values = new List<double>();
            _size = size <= minSize ? 100 : size;
            _minSize = minSize <= 2 ? 20 : minSize;
        }
        #endregion

        #region Public Functions
        /// <summary>
        /// Adds a value to the TrainingWindow.
        /// </summary>
        /// <param name="value">Value to be added.</param>
        public void Add(double value)
        {
            _values.Add(value);
            _sum += value;

            // if the size of the window is going to be exceeded, remove the first item from the list.
            if (_values.Count > _size)
            {
                _sum -= _values[0];
                _values.RemoveAt(0);
            }
        }

        /// <summary>
        /// The avergae is based on the sum of all values divided by the count.
        /// </summary>
        /// <returns>Returns the average of the TrainingWindow's values. Returns -1 if the minimum number of values is not reached.</returns>
        public double GetAverage()
        {
            if (_values.Count < _minSize)
            {
                return 0;
            }
            else
            {
                return _sum / _values.Count;
            }
        }

        /// <summary>
        /// Removes all values.
        /// </summary>
        public void Reset()
        {
            this._values = new List<double>();
            this._sum = 0;
        }
        #endregion
    }
}