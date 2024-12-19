using System.Numerics;
using ConvNetSharp.V2.Graph;
using ConvNetSharp.V2.Ops;

namespace ConvNetSharp.V2;

/// <summary>
///     Represents a session for managing the execution, differentiation, and scope of computation graphs in ConvNetSharp.
///     Handles the lifecycle of graph operations, placeholders, and variables.
/// </summary>
/// <typeparam name="T">The numeric type used in computations (e.g., float, double).</typeparam>
public class Session<T> : IDisposable where T : struct, INumber<T>, IMinMaxValue<T>
{
    private readonly ConvNetSharp<T> _convNetSharp; // Dependency for scope management and graph operations
    private bool _derivativeComputed; // Tracks whether differentiation has been performed
    private bool _disposed; // Tracks whether the session has been disposed

    /// <summary>
    ///     Initializes a new instance of the <see cref="Session{T}" /> class.
    ///     If no active ConvNetSharp instance exists, it creates one.
    /// </summary>
    public Session() : this(ConvNetSharp<T>.Current ?? new ConvNetSharp<T>())
    {
        if (ConvNetSharp<T>.Current == null)
        {
            ConvNetSharp<T>.SetActive(this._convNetSharp);
        }
    }

    public Session(ConvNetSharp<T> convNetSharp)
    {
        this._convNetSharp = convNetSharp ?? throw new ArgumentNullException(nameof(convNetSharp));
    }


    /// <summary>
    ///     Gets the current cost function for the session.
    /// </summary>
    public Op<T> Cost { get; private set; }

    /// <summary>
    ///     Gets or sets the learnable variables in the computation graph.
    /// </summary>
    public Dictionary<string, Variable<T>> LearnableVariables { get; set; } = new();

    /// <summary>
    ///     Gets or sets the current step of the session.
    /// </summary>
    public long Step { get; set; }

    /// <summary>
    ///     Gets the batch size of the data being processed.
    /// </summary>
    public int BatchSize { get; private set; }

    /// <summary>
    ///     Releases all resources used by the session and disposes of the computation graph.
    /// </summary>
    public void Dispose()
    {
        if (this._disposed)
        {
            return;
        }

        // Dispose the computation graph starting from the Cost node.
        // This assumes Cost is connected to all other nodes, which may not always be the case.
        Op<T>.DisposeGraph(this.Cost);

        this._disposed = true;
    }

    /// <summary>
    ///     Creates a scoped context for grouping operations under a named scope.
    ///     Delegates scope management to the ConvNetSharp instance.
    /// </summary>
    /// <param name="name">The name of the scope.</param>
    /// <returns>A disposable ScopedContext object.</returns>
    public ScopedContext<T> Scope(string name)
    {
        return this._convNetSharp.Scope(name);
    }

    /// <summary>
    ///     Performs automatic differentiation using reverse accumulation.
    /// </summary>
    /// <param name="cost">The cost function to differentiate.</param>
    /// <param name="gradient">The initial gradient. Defaults to 1 if not specified.</param>
    public void Differentiate(Op<T> cost, Op<T> gradient = null)
    {
        if (this._derivativeComputed)
        {
            return;
        }

        // Clear existing derivatives
        var visitor = new OpVisitor<T>(op => { op.Derivative = null; });
        cost.Accept(visitor);

        this.Cost = cost;

        // Initialize gradient for the cost node
        cost.Derivative = gradient ?? ConvNetSharp<T>.One;

        // Perform reverse accumulation
        var differentiateVisitor = new DifferentiateVisitor<T>();
        cost.Accept(differentiateVisitor);

        // Ensure all learnable variables have valid derivatives
        foreach (var variable in this.LearnableVariables.Values)
        {
            if (variable.Derivative == null)
            {
                Console.WriteLine($"Initializing gradient for learnable variable: {variable.Name}");
                variable.Differentiate();
            }
        }

        this._derivativeComputed = true;
    }


    /// <summary>
    ///     Dumps the computation graph to a file for debugging purposes.
    /// </summary>
    /// <param name="fun">The root node of the graph.</param>
    /// <param name="fileName">The file to write the graph structure and results.</param>
    public void Dump(Op<T> fun, string fileName)
    {
        using var sw = new StreamWriter(File.Create(fileName));

        var streamWriter = sw;
        var visitor = new OpVisitor<T>(op =>
        {
            streamWriter.WriteLine(op);
            streamWriter.WriteLine(op.Result == null ? "[Null]" : op.Result.ToString());
        });
        fun.Accept(visitor);
    }

    /// <summary>
    ///     Finds a variable by name in the computation graph.
    /// </summary>
    /// <param name="fun">The root node of the graph to search.</param>
    /// <param name="name">The name of the variable.</param>
    /// <returns>The variable operation if found; otherwise, null.</returns>
    public Op<T> GetVariableByName(Op<T> fun, string name)
    {
        Op<T> result = null;

        var visitor = new OpVisitor<T>(op =>
        {
            if (op is INamedOp<T> variable && variable.Name == name)
            {
                result = op;
            }
        });

        fun.Accept(visitor);

        return result;
    }

    /// <summary>
    ///     Initializes placeholders in the computation graph with provided values.
    /// </summary>
    /// <param name="fun">The root node of the graph.</param>
    /// <param name="dictionary">A dictionary mapping placeholder names to volumes.</param>
    public void InitializePlaceHolders(Op<T> fun, Dictionary<string, Volume<T>> dictionary)
    {
        // Determine batch size from the provided data.
        this.BatchSize = dictionary.Values.Select(o => o.Shape.Dimensions[3]).Max();

        this.UpdatePlaceHolder(fun, dictionary);
    }

    /// <summary>
    ///     Evaluates the computation graph with the provided input values.
    /// </summary>
    /// <param name="fun">The root node of the graph.</param>
    /// <param name="dictionary">A dictionary mapping placeholder names to volumes.</param>
    /// <param name="incrementStep">Indicates whether to increment the step counter.</param>
    /// <returns>The result of the graph evaluation.</returns>
    public Volume<T> Run(Op<T> fun, Dictionary<string, Volume<T>> dictionary, bool incrementStep = true)
    {
        if (dictionary != null && dictionary.Any())
        {
            this.BatchSize = dictionary.Values.Select(o => o.Shape.Dimensions[3]).Max();
            this.UpdatePlaceHolder(fun, dictionary);
        }

        var result = fun.Evaluate(this);

        if (incrementStep)
        {
            this.Step++;
        }

        return result;
    }

    /// <summary>
    ///     Updates the values of placeholders in the computation graph.
    /// </summary>
    /// <param name="fun">The root node of the graph.</param>
    /// <param name="dictionary">A dictionary mapping placeholder names to volumes.</param>
    public void UpdatePlaceHolder(Op<T> fun, Dictionary<string, Volume<T>> dictionary)
    {
        var visitor = new OpVisitor<T>(op =>
        {
            if (op is PlaceHolder<T> placeHolder)
            {
                if (!dictionary.TryGetValue(placeHolder.Name, out var volume))
                {
                    throw new Exception($"Cannot find key '{placeHolder.Name}' in the provided dictionary");
                }

                placeHolder.SetValue(volume);
            }

            if (op is Variable<T> { IsLearnable: true } variable)
            {
                this.LearnableVariables[variable.Name] = variable;
            }
        });

        fun.Accept(visitor);
    }
}