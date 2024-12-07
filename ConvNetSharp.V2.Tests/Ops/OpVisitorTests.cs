using ConvNetSharp.V2.Ops;
using NUnit.Framework;

namespace ConvNetSharp.V2.Tests.Ops;

[TestFixture]
public class OpVisitorTests
{
    private class TestOp(ConvNetSharp<double> graph) : Op<double>(graph)
    {
        public override string Representation => "TestOp";

        public override void Differentiate()
        {
            throw new NotImplementedException();
        }

        public override Volume<double> Evaluate(Session<double> session)
        {
            return Result;
        }
    }

    [Test]
    public void Visit_ShouldApplyFunctionToEachNode()
    {
        // Arrange
        var graph = new ConvNetSharp<double>();
        var op1 = new TestOp(graph);
        var op2 = new TestOp(graph);
        var op3 = new TestOp(graph);

        op1.AddParent(op2);
        op2.AddParent(op3);

        var visitedNodes = new HashSet<Op<double>>();

        void Func(Op<double> node)
        {
            visitedNodes.Add(node);
        }

        var visitor = new OpVisitor<double>(Func);

        // Act
        visitor.Visit(op1);

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(visitedNodes, Contains.Item(op1), "op1 should be visited.");
            Assert.That(visitedNodes, Contains.Item(op2), "op2 should be visited.");
            Assert.That(visitedNodes, Contains.Item(op3), "op3 should be visited.");
        });
    }

    [Test]
    public void Visit_WithBothWays_ShouldVisitParentsAndChildren()
    {
        // Arrange
        var graph = new ConvNetSharp<double>();
        var op1 = new TestOp(graph);
        var op2 = new TestOp(graph);
        var op3 = new TestOp(graph);

        op1.AddParent(op2);
        op2.AddParent(op3);
        op3.AddParent(op1); // Creates a bidirectional link

        var visitedNodes = new HashSet<Op<double>>();

        void Func(Op<double> node)
        {
            visitedNodes.Add(node);
        }

        var visitor = new OpVisitor<double>(Func, true);

        // Act
        visitor.Visit(op1);

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(visitedNodes, Contains.Item(op1), "op1 should be visited.");
            Assert.That(visitedNodes, Contains.Item(op2), "op2 should be visited.");
            Assert.That(visitedNodes, Contains.Item(op3), "op3 should be visited.");
        });
    }

    [Test]
    public void Visit_ShouldNotRevisitNodes()
    {
        // Arrange
        var graph = new ConvNetSharp<double>();
        var op1 = new TestOp(graph);
        var op2 = new TestOp(graph);
        var op3 = new TestOp(graph);

        op1.AddParent(op2);
        op2.AddParent(op3);
        op3.AddParent(op1); // Creates a cycle

        var visitCount = 0;

        void Func(Op<double> node)
        {
            visitCount++;
        }

        var visitor = new OpVisitor<double>(Func);

        // Act
        visitor.Visit(op1);

        // Assert
        Assert.That(visitCount, Is.EqualTo(3), "Nodes should not be revisited even in cyclic graphs.");
    }

    [Test]
    public void Visit_ShouldThrowException_WhenFuncIsNull()
    {
        // Arrange
        var graph = new ConvNetSharp<double>();
        var op = new TestOp(graph);

        // Act & Assert
        Assert.That(() => new OpVisitor<double>(null!), Throws.ArgumentNullException);
    }

    [Test]
    public void Visit_ShouldHandleEmptyGraphGracefully()
    {
        // Arrange
        var visitor = new OpVisitor<double>(_ => { });

        // Act & Assert
        Assert.That(() => visitor.Visit(null), Throws.Nothing, "Visiting a null graph should not throw.");
    }
}