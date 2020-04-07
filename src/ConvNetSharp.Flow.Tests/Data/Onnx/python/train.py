import os
from models import ConvModel
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

batch_size = 256

# Data
train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor(), download=True)
test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model
model = ConvModel()

# Learning
sgd = SGD(model.parameters(), lr=1e-1)
cross_error = CrossEntropyLoss()

n_epoch = 5
for epoch in range(n_epoch):
    for idx, (train_x, train_label) in enumerate(train_loader):
        label_np = np.zeros((train_label.shape[0], 10))
        sgd.zero_grad()
        predict_y = model(train_x.float())
        error = cross_error(predict_y.squeeze(), train_label.long())
        if idx % 10 == 0:
            print(f'idx: {idx}, error: {error}')
        error.backward()
        sgd.step()

    correct = 0
    sum = 0

    for idx, (test_x, test_label) in enumerate(test_loader):
        predict_y = model(test_x.float()).detach()
        predict_ys = np.argmax(predict_y.squeeze(), axis=-1)
        label_np = test_label.numpy()
        _ = predict_ys == test_label
        correct += np.sum(_.numpy(), axis=-1)
        sum += _.shape[0]

    print(f'accuracy: {correct / sum:.2f}')
    os.makedirs('models', exist_ok=True)
    torch.save({'state_dict': model.state_dict()}, f'models/checkpoint_{correct / sum}.pth.tar')

# Export the model
batch_size = 1
x = torch.randn(batch_size, 1, 28, 28)
torch_out = model(x)

torch.onnx.export(model,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  "../mnist.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                'output': {0: 'batch_size'}})
