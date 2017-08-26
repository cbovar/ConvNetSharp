using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers.Double;
using System;
using System.Windows.Forms;

namespace SlotMachineDemo
{
    internal class Program
    {
        #region Main
        [STAThread]
        static void Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new TrainingForm());
        }
        #endregion
    }
}