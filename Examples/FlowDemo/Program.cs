using System;

namespace FlowDemo
{
    internal class Program
    {
        [STAThread]
        private static void Main()
        {
            //ExampleCpuSingle.Example1();
            //ExampleCpuDouble.Example2();
            //ExampleGpuSingle.Example1();
            NetExampleSingle.Classify2DDemo();
        }
    }
}