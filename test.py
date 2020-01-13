import numpy as np
import torch as t
import torch.nn as nn

x = np.array([0.87, 0.23, 0.65, 0.99, 0.003, -0.12, 1.3])
y = np.array([1., 0., 1., 1., 0., 0., 1.])

x2 = np.array([0.12, 0.43, 0.789, 0.63, 0.213, -0.34, 1.4])
y2 = np.array([0., 1., 1., 1., 0., 0., 1.])

xt = t.from_numpy(x)
yt = t.from_numpy(y)

x2t = t.from_numpy(x2)
y2t = t.from_numpy(y2)

true_loss = nn.BCEWithLogitsLoss()

print("##########################################")
print(f"The value we want is: {true_loss(xt, yt) + true_loss(x2t, y2t)}")
print("##########################################")


def torch_loss(input, target):
    max_val = input.clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    return loss.mean()


print("\n##########################################")
print(f"TORCH Loss is: {torch_loss(xt, yt) + torch_loss(x2t, y2t)}")
print("##########################################")


def my_loss(x, y):
    max_val = np.clip(x, 0, None)
    loss = x - x * y + max_val + np.log(np.exp(-max_val) + np.exp((-x - max_val)))
    return loss.mean()


print("\n##########################################")
print(f"MY Loss is: {my_loss(x, y) + my_loss(x2t, y2t)}")
print("##########################################")