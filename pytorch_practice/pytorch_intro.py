from __future__ import print_function
import numpy as np
import torch

x = torch.empty(5, 3)
print(x)
x = torch.rand(5, 3)
print(x)
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
x = torch.tensor([5.5, 3])
print(x)
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)
print(x.size())

# Addition
print('')
print('Addition')
y = torch.rand(5, 3)
print(y)
print(x + y)
print(torch.add(x, y))
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
y.add_(x)
print(y)
print(y[:, 0])

x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item())

# Torch Tensor --> Numpy
print('')
print('Torch Tensor --> Numpy')
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

# Numpy --> Torch Tensor
print('')
print('Numpy --> Torch Tensor')
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# let us run this cell only if CUDA is available (it's not for my desktop computer)
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)  # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  # ``.to`` can also change dtype together!
