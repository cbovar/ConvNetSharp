using System;
using System.Windows.Data;
using System.Windows.Media;
using QuickGraph;

namespace ConvNetSharp.Utils.GraphVisualizer
{
    public class OpVertex
    {
        public string Name { get; set; }

        public string Shape { get; set; }

        public string Type { get; set; }

        public Color Color { get; set; }
    }

    public class OpEdge : Edge<object>
    {
        public OpEdge(OpVertex source, OpVertex target) : base(source, target)
        {
        }
    }

    public class EdgeColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return new SolidColorBrush((Color)value);
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}