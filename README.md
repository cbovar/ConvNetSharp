# Work in Progress

## Computation graph

- It is a graph of operations (Ops) representing a function.
e.g.
```c#
	var x = cns.PlaceHolder("x");	
	var y = cns.PlaceHolder("y");
	
	var W = cns.Variable(1.0, "W");
	var b = cns.Variable(2.0, "b");
	
	var fun = x * W + b; // Graph
```
- Easier to create new layers and to experiment
- Auto differentiation of the graph
- Any loss function. Loss is distinct from layers.
- Easier memory management: Ops know what to allocate for result or temporary objects. Released when graph is disposed
- We can still keep ConvNetSharp/JS layers: layers will just extend the computation graph. Most users can only use layers. Advanced users can use lower level Ops.
- Looks familiar for people who already know TensorFlow

## Tools

- Display graph (with GraphSharp in WPF)

![Computation Graph](https://github.com/cbovar/ConvNetSharp/blob/Develop/img/computationGraph.png)


## TODO:

- A lot!
- Implement usual ConvNetSharp layers using computation graph internaly
- More Ops (CPU and GPU)
- Optimizers
- Scope handling (to gather Ops that belong to a functional group, to have nodes with the same name)
- Serialization (not sure how it is done in TF but we can separate graph and data)
- Explore the use of cuDNN cudnnOpTensor
- Migrate to cuDNN v6 ([ManagedCuda](https://github.com/kunzmi/managedCuda) upgraded a few days ago)
- Optimization of graph ?
- VS extensions to help debug (shows graph, value of node, et.)
