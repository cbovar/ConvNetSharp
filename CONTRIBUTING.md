# How to contribute

1) Pick or create some issues and assign them to yourself
2) Create a feature branch
3) Commit documented changes along with unit tests
4) Create a Pull request


Contribution is welcome on any aspect of the project but the following points are the priority:
* Better documentation: Fill up missing public methods documentation headers, find a way to easily export it and host it in github
* Better error messages: End user should understand why an exception is raised (e.g. wrong input shape)
* Make the api less verbose (e.g. volume creation is ugly: ```BuilderInstance.Volume.From(new[] { 0.3, -0.5 }, new Shape(2))```)
* Optimization of ConvNetSharp.Flow computation graph (factorization, simple optimization like replace X * 1 by X, ...)
