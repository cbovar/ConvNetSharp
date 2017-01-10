using System;
using System.Collections.Generic;

namespace FluentMnistDemo
{
    public class CircularBuffer<T>
    {
        private readonly T[] buffer;
        private int nextFree;

        public CircularBuffer(int capacity)
        {
            this.Capacity = capacity;
            this.Count = 0;
            this.buffer = new T[capacity];
        }

        public int Capacity { get; private set; }

        public int Count { get; private set; }

        public IEnumerable<T> Items
        {
            get { return this.buffer; }
        }

        public void Add(T o)
        {
            this.buffer[this.nextFree] = o;
            this.nextFree = (this.nextFree + 1)%this.buffer.Length;
            this.Count = Math.Min(this.Count + 1, this.Capacity);
        }
    }
}