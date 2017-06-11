using System;

namespace ConvNetSharp.Example
{
    internal class Program
    {
        [STAThread]
        private static void Main()
        {
            ExampleCpuSingle.Example1();
            //ExampleCpuDouble.Example1();
            //ExampleGpuSingle.Example1();
            //NetExampleSingle.Example1();
        }
    }
}