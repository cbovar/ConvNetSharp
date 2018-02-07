This example shows how to train a neural network to classify a two dimensions vector (x1, x2) into one of two classes (y).
This is therefore supervised training (because the label is given in the training set)

We want to learn `(x1, x2) -> y where y = 0 OR y = 1`

### Trainin set

| x1  | x2 | y (label / class) |
| ------------- | ------------- | ------------- |
|-0.4326| 1.1909| 1|
|3.0| 4.0|1|
|0.1253| -0.0376|1|
|0.2877| 0.3273|1|
|-1.1465| 0.1746|1|
|1.8133| 1.0139|0|
|2.7258| 1.0668|0|
|1.4117| 0.5593|0|
|4.1832| 0.3044|0|
|1.8636| 0.1677|0|
|0.5| 3.2|1|
|0.8| 3.2|1|
|1.0| -2.2|1|

### Code

#### Learning 
This example shows how to input a batch of data (**n** examples at the time) rather than one example at the time.
This means that the network will learn from several example at the time rather than learning by looking at one example at the time. This is more efficient.

We first try to teach the network `(x1, x2) -> (y1, y2)` where `(y1, y2)` is the one-hot encoded label.  
```y = 1 -> (y1, y2) = (1, 0)
y = 2 -> (y1, y2) = (0, 1)
```

In *Classify2DUpdate*, a Volume of shape [1, 1, 2, n] is created to store input data `(x1, x2)` and another Volume of shape [1, 1, 2, n] is created to store the one-hot encoded labels where n is the batchsize.

#### Inference

An input volume **netx** of shape [1, 1, 2, n] is created. For each input (n inputs here), we want to guess the associated label.
This is done with those lines:

```c#
var result = net.Forward(netx);
var c = net.GetPrediction();
```
**Result** will contain the one-hot encoded predicted labels
**GetPrediction** is a convenience method that will translate from one-hot encoded labels to the labels.
```
e.g.  
(1, 0) -> 0
(0, 1) -> 1
```
*note:* Learning was done using batch of **n** elements but you can do the inference on any batch size (i.e. the 'n' of inference doesnt have to be the same as the 'n' used during training)
