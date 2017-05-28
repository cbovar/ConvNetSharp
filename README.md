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
- Easier memory management: Ops know what to allocate for result or temporary objects. Release when graph is disposed
- We can still keep ConvNetSharp/JS layers: layers will just extend the computation graph. Most users can use layers only but
- Looks familiar for people who already know TensorFlow

## Tools

- Display graph (with GraphSharp in WPF)

![Computation Graph](https://github.com/cbovar/ConvNetSharp/tree/Develop/img/computationGraph)
