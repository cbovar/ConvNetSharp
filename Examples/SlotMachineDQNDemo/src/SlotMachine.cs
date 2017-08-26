namespace SlotMachineDemo
{
    public class SlotMachine
    {
        #region Member Fields
        // Reels
        private Reel _reelLeft;
        private Reel _reelMid;
        private Reel _reelRight;

        // Scores and rewards
        // Order: Cherry, plum, melon, lemon, orange, seven
        private readonly double[] _ITEM_VALUES = new double[] { 0.05 * 2, 0.1 * 2, 0.2 * 2, 0.35 * 2, 0.4 * 2, 1 * 2 };
        private const double _HIT_REWARD = 2.5;
        private const double _HIT_PUNISH = -0.3;
        #endregion

        #region Member Properties
        /// <summary>
        /// If at least one reel is spinning, then this indicates that the slot machine is running.
        /// </summary>
        public bool IsRunning
        {
            get; set;
        }

        /// <summary>
        /// Read-only. Left reel.
        /// </summary>
        public Reel ReelLeft
        {
            get { return _reelLeft; }
        }

        /// <summary>
        /// Read-only. Mid reel.
        /// </summary>
        public Reel ReelMid
        {
            get { return _reelMid; }
        }

        /// <summary>
        /// Read-only. Right reel.
        /// </summary>
        public Reel ReelRight
        {
            get { return _reelRight; }
        }

        /// <summary>
        /// Score is 0 if the slot machine is still running. As soon as it is done, the result is assigned.
        /// </summary>
        public double Score
        {
            get; set;
        }
        #endregion

        #region Constructor
        /// <summary>
        /// Initializes a slot machine using 3 reels.
        /// </summary>
        public SlotMachine()
        {
            IsRunning = false;
            _reelLeft = new Reel(1);
            _reelMid = new Reel(2);
            _reelRight = new Reel(1);
        }
        #endregion

        #region Public Functions
        /// <summary>
        /// Updates the slot machine's state.
        /// </summary>
        public void Tick()
        {
            if (IsRunning)
            {
                _reelLeft.Tick();
                _reelMid.Tick();
                _reelRight.Tick();
            }
        }

        /// <summary>
        /// Commences the slot machine behavior.
        /// </summary>
        public void Start()
        {
            Score = 0;
            IsRunning = true;
            _reelLeft.Start();
            _reelMid.Start();
            _reelRight.Start();
        }

        /// <summary>
        /// Stops one reel (order: left, mid, right)
        /// </summary>
        /// <returns>Based on the stopped reel, a reward is returned.</returns>
        public double StopReel()
        {
            if(_reelLeft.IsSpinning)
            {
                _reelLeft.Stop();
                return _ITEM_VALUES[(int)_reelLeft.CurrentItem];
            }
            else if(_reelMid.IsSpinning)
            {
                _reelMid.Stop();
                if(_reelLeft.CurrentItem.Equals(_reelMid.CurrentItem))
                {
                    return _HIT_REWARD;
                }
                else
                {
                    return _HIT_PUNISH;
                }
            }
            else if(_reelRight.IsSpinning)
            {
                _reelRight.Stop();
                FinishSlotMachine();
                if (_reelLeft.CurrentItem.Equals(_reelMid.CurrentItem) && _reelLeft.CurrentItem.Equals(_reelRight.CurrentItem))
                {
                    return _HIT_REWARD;
                }
                else
                {
                    return _HIT_PUNISH;
                }
            }
            else
            {
                return 0;
            }
        }
        #endregion

        #region Private Functions
        /// <summary>
        /// Concludes the slot machine and processes the result.
        /// </summary>
        private void FinishSlotMachine()
        {
            EvaluateResult();
            IsRunning = false;
        }

        private void EvaluateResult()
        {
            // Check if all reels share the same item on their middle slot.
            if(_reelLeft.CurrentItem == _reelMid.CurrentItem && _reelLeft.CurrentItem  == _reelRight.CurrentItem)
            {
                Score = _ITEM_VALUES[(int)_reelLeft.CurrentItem];
            }
            else
            {
                Score = 0;
            }
        }
        #endregion
    }
}