namespace TestGPU
{

    struct SomeBasicType
    {
        // yes, these are public mutable fields; we are explicitly **not**
        // trying to provide abstractions here - we're holding our hands
        // up and saying "you're playing with raw memory, don't screw up"
        public int Id;
        public uint Value;
    }
}
