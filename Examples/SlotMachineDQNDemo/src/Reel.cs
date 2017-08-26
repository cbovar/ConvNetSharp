using System;

namespace SlotMachineDemo
{
    public class Reel
    {
        #region Member Fields
        private int _itemTickDuration = 3;
        private Item[] _slots = new Item[3];
        private Random _random;
        private double[] _probabilityDisitribution = new double[] { 0.3, 0.25, 0.2, 0.1, 0.1, 0.05}; // Order: Cherry, Plum, WaterMelon, Lemon, Orange, Seven
        private int _tickCounter = 0;
        #endregion

        #region Member Properties
        /// <summary>
        /// Returns the item of the middle slot.
        /// </summary>
        public Item CurrentItem
        {
            get { return _slots[1]; } // Middle slot
        }

        /// <summary>
        /// Returns the slots' items.
        /// </summary>
        public Item[] SlotItems
        {
            get { return _slots; }
        }

        /// <summary>
        /// Indicates if the reel is spinning or not.
        /// </summary>
        public bool IsSpinning
        {
            get; set;
        }
        #endregion

        #region Constructor
        /// <summary>
        /// Constructs a reel based of 3 slots.
        /// </summary>
        /// <param name="itemTickDuration">The duration before switchting the slots.</param>
        public Reel(int itemTickDuration)
        {
            _itemTickDuration = itemTickDuration;
            _random = new Random();
            InitializeSlots();
        }
        #endregion

        #region Public Functions
        /// <summary>
        /// Updates the reel.
        /// </summary>
        public void Tick()
        {
            if (IsSpinning)
            {
                _tickCounter++;
                if(_tickCounter >= _itemTickDuration)
                {
                    UpdateSlots();
                }
            }
        }

        /// <summary>
        /// Starts the reel by making it "spinning".
        /// </summary>
        public void Start()
        {
            InitializeSlots();
            IsSpinning = true;
        }

        /// <summary>
        /// Stops the reel's "spinning".
        /// </summary>
        public void Stop()
        {
            IsSpinning = false;
            _tickCounter = 0;
        }
        #endregion

        #region Private Functions
        /// <summary>
        /// 
        /// </summary>
        /// <returns>Returns an item based on the given probability distribution.</returns>
        private Item SampleItem()
        {
            double rand = _random.NextDouble();
            if(rand <= _probabilityDisitribution[0])
            {
                return Item.Cherry;
            }
            else if (rand <= _probabilityDisitribution[0] + _probabilityDisitribution[1])
            {
                return Item.Plum;
            }
            else if(rand <= _probabilityDisitribution[0] + _probabilityDisitribution[1] + _probabilityDisitribution[2])
            {
                return Item.Melon;
            }
            else if (rand <= _probabilityDisitribution[0] + _probabilityDisitribution[1] + _probabilityDisitribution[2] + _probabilityDisitribution[3])
            {
                return Item.Lemon;
            }
            else if (rand <= _probabilityDisitribution[0] + _probabilityDisitribution[1] + _probabilityDisitribution[2] + _probabilityDisitribution[3] + _probabilityDisitribution[4])
            {
                return Item.Orange;
            }
            else
            {
                return Item.Seven;
            }
        }

        /// <summary>
        /// Initializes the slots with items.
        /// </summary>
        private void InitializeSlots()
        {
            _slots[0] = SampleItem(); // Upper Slot
            _slots[1] = SampleItem(); // Middle Slot
            _slots[2] = SampleItem(); // Bottom Slot
        }

        /// <summary>
        /// Shifts the slot items downwards. The upper slot samples a new item.
        /// </summary>
        private void UpdateSlots()
        {
            _slots[2] = _slots[1]; // Bottom Slot
            _slots[1] = _slots[0]; // Middle Slot
            _slots[0] = SampleItem(); // Upper Slot
        }
        #endregion
    }
}