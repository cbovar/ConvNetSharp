using System;
using System.Collections.Generic;
using System.Windows.Media;
using ConvNetSharp.Flow.Ops;
using QuickGraph;

namespace ConvNetSharp.Utils.GraphVisualizer
{
    public class ViewModel<T> where T : struct, IEquatable<T>, IFormattable
    {
        public ViewModel(Op<T> root)
        {
            var colors = CreateColorPalette(20);

            var set = new HashSet<Op<T>>();

            var visitor = new OpVisitor<T>(op =>
            {
                if (!set.Contains(op))
                {
                    set.Add(op);
                }
            });
            root.Accept(visitor);

            var graph = new BidirectionalGraph<object, IEdge<object>>(); // new OpGraph();

            var dico = new Dictionary<Op<T>, OpVertex>();
            foreach (var op in set)
            {
                var colIndex = op.GetType().GetHashCode() % colors.Count;
                var opVertex = new OpVertex
                {
                    Name = op.Representation,
                    Color = colors[colIndex],
                    Shape = op.Result?.Shape.ToString() != null ? "["+op.Result.Shape+"]" : string.Empty
                };
                dico[op] = opVertex;
                graph.AddVertex(opVertex);
            }

            foreach (var op in set)
            {
                foreach (var parent in op.Parents)
                {
                    graph.AddEdge(new OpEdge(dico[parent], dico[op]));
                }
            }

            this.Graph = graph;
        }

        public BidirectionalGraph<object, IEdge<object>> Graph { get; set; }

        /// <summary>
        ///     from https://stackoverflow.com/questions/20098828/create-a-nice-color-palette
        /// </summary>
        /// <param name="interval"></param>
        private static List<Color> CreateColorPalette(int interval)
        {
            var colors = new List<Color>();

            for (var red = 0; red < 255; red += interval)
            {
                for (var green = 0; green < 255; green += interval)
                {
                    for (var blue = 0; blue < 255; blue += interval)
                    {
                        if ((red < 80) | (blue < 80) | (green < 80))
                        {
                            colors.Add(Color.FromArgb(255, (byte) red, (byte) green, (byte) blue));
                        }
                    }
                }
            }

            return colors;
        }
    }
}