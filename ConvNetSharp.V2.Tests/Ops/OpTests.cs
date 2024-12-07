using ConvNetSharp.V2.Ops;
using NUnit.Framework;

namespace ConvNetSharp.V2.Tests.Ops;

[TestFixture]
public class OpTests
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
    public void AddParent_ShouldAddParentAndUpdateChildren()
    {
        // Arrange
        var graph = new ConvNetSharp<double>();
        var op1 = new TestOp(graph);
        var op2 = new TestOp(graph);

        // Act
        op1.AddParent(op2);

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(op1.Parents, Does.Contain(op2), "op1 should have op2 as a parent.");
            Assert.That(op2.Children, Does.Contain(op1), "op2 should have op1 as a child.");
        });
    }

    [Test]
    public void RemoveParent_ShouldRemoveParentAndUpdateChildren()
    {
        // Arrange
        var graph = new ConvNetSharp<double>();
        var op1 = new TestOp(graph);
        var op2 = new TestOp(graph);
        op1.AddParent(op2);

        // Act
        op1.RemoveParent(op2);

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(op1.Parents, Does.Not.Contain(op2), "op1 should not have op2 as a parent after removal.");
            Assert.That(op2.Children, Does.Not.Contain(op1), "op2 should not have op1 as a child after removal.");
        });
    }

    [Test]
    public void SetDirty_ShouldMarkChildrenAsDirty()
    {
        // Arrange
        var graph = new ConvNetSharp<double>();
        var parent = new TestOp(graph);
        var child = new TestOp(graph);
        parent.AddParent(child);

        // Act
        parent.SetDirty();

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(parent.IsDirty, Is.True, "Parent should be marked as dirty.");
            Assert.That(child.IsDirty, Is.True, "Child should be marked as dirty.");
        });
    }

    [Test]
    public void DisposeGraph_ShouldDisposeNodes()
    {
        // Arrange
        var graph = new ConvNetSharp<double>();
        var op = new TestOp(graph);

        // Act & Assert
        Assert.That(() => Op<double>.DisposeGraph(op), Throws.Nothing, "DisposeGraph should not throw any exceptions.");
    }

    [Test]
    public void OperatorPlus_ShouldCombineTwoOps()
    {
        // Arrange
        var graph = new ConvNetSharp<double>();
        var op1 = new TestOp(graph);
        var op2 = new TestOp(graph);

        // Act
        var result = op1 + op2;

        // Assert
        Assert.That(result, Is.InstanceOf<Add<double>>());
    }

    [Test]
    public void OperatorMinus_ShouldSubtractTwoOps()
    {
        // Arrange
        var graph = new ConvNetSharp<double>();
        var op1 = new TestOp(graph);
        var op2 = new TestOp(graph);

        // Act
        var result = op1 - op2;

        // Assert
        Assert.That(result, Is.InstanceOf<Add<double>>(), "Subtraction is implemented as addition with negation.");
    }

    [Test]
    public void OperatorMultiply_ShouldMultiplyTwoOps()
    {
        // Arrange
        var graph = new ConvNetSharp<double>();
        var op1 = new TestOp(graph);
        var op2 = new TestOp(graph);

        // Act
        var result = op1 * op2;

        // Assert
        Assert.That(result, Is.InstanceOf<Mult<double>>());
    }

    [Test]
    public void OperatorNegate_ShouldNegateAnOp()
    {
        // Arrange
        var graph = new ConvNetSharp<double>();
        var op = new TestOp(graph);

        // Act
        var result = -op;

        // Assert
        Assert.That(result, Is.InstanceOf<Negate<double>>());
    }

    [Test]
    public void OperatorImplicit_ShouldCreateConstOp()
    {
        // Act
        var result = (Op<double>)5.0;

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(result, Is.InstanceOf<Const<double>>());
            Assert.That(((Const<double>)result).Value, Is.EqualTo(5.0));
        });
    }

    [Test]
    public void RegisterDerivative_ShouldStoreDerivative()
    {
        // Arrange
        var graph = new ConvNetSharp<double>();
        var op = new TestOp(graph);
        var derivative = new TestOp(graph);

        // Act
        op.RegisterDerivative(derivative);

        // Assert
        Assert.That(op.Derivative, Is.SameAs(derivative));
    }

    [Test]
    public void RegisterDerivative_ShouldCombineDerivatives()
    {
        // Arrange
        var graph = new ConvNetSharp<double>();
        var op = new TestOp(graph);
        var derivative1 = new TestOp(graph);
        var derivative2 = new TestOp(graph);

        // Act
        op.RegisterDerivative(derivative1);
        op.RegisterDerivative(derivative2);

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(op.Derivative, Is.Not.Null);
            Assert.That(op.Derivative, Is.InstanceOf<Add<double>>());
        });
    }
}