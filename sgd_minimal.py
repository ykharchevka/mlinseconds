import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Let's make some data for a linear regression.
A = 3.1415926
b = 2.7189351
error = 0.1
N = 100 # number of data points

# Data
X = Variable(torch.randn(N, 1))

# (noisy) Target values that we want to learn.
t = A * X + b + Variable(torch.randn(N, 1) * error)

# Creating a model, making the optimizer, defining loss
model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, dampening=0.9)
loss_fn = nn.MSELoss()

# Run training
niter = 5
for _ in range(0, niter):
	optimizer.zero_grad()
	predictions = model(X)
	loss = loss_fn(predictions, t)
	loss.backward()
	optimizer.step()

	print("-" * 50)
	print("error = {}".format(loss.data.item()))
	print("learned A = {}".format(list(model.parameters())[0].data[0, 0]))
	print("learned b = {}".format(list(model.parameters())[1].data[0]))
