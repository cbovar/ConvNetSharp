using System;
using System.Collections.Generic;

namespace MnistDemo
{
    public class CircularBuffer<T>(int capacity)
    {
        private readonly T[] _buffer = new T[capacity];
        private int _nextFree;

        public int Capacity { get; } = capacity;

        public int Count { get; private set; } = 0;

        public IEnumerable<T> Items => this._buffer;

        public void Add(T o)
        {
            this._buffer[this._nextFree] = o;
            this._nextFree = (this._nextFree + 1) % this._buffer.Length;
            this.Count = Math.Min(this.Count + 1, this.Capacity);
        }
    }
}