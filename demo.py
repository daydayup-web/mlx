import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from sympy import false

class SimpleNetWithBMM(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim1, output_dim2):
        super(SimpleNetWithBMM, self).__init__()
        self.fc1 = nn.Linear(input_dim1 * input_dim2, output_dim1 * output_dim2, false)
        print("Weight shape:", self.fc1.weight.shape)
        print(self.fc1.weight.tolist())

    def __call__(self, x):
        out = self.fc1(x)
        return out

input_size1 = 64
input_size2 = 32
output_size1 = 64
output_size2 = 10

model = SimpleNetWithBMM(input_size1, input_size2, output_size1, output_size2)
print(model)

batch_size = 1
x = mx.random.uniform(-1, 1, shape=(batch_size, input_size1 * input_size2))
print("Input shape:", x.shape)
print(x.tolist())

# 标签
eps = 1e-2 * mx.random.normal((batch_size, output_size1 * output_size2))
print("eps shape:", eps.shape)
target = x @ mx.transpose(model.fc1.weight) + eps
print("Target shape:", target.shape)
print(target.tolist())

# 定义损失函数
def loss(models, inputs, targets):
    pred = models(inputs)
    print("Output shape:", pred.shape)
    print(pred.tolist())
    ce = nn.losses.mse_loss(pred, targets)
    return ce.mean()

# 定义优化器
learning_rate = 0.01
optimizer = optim.Adam(learning_rate)

# 前向传播&反向传播
loss_value_and_grad = nn.value_and_grad(model, loss)
loss, grad = loss_value_and_grad(model, x, target)
print("Loss shape:", loss.shape)
print("Loss:", loss)
print("Grad shape:", grad['fc1']['weight'].shape)
print("Grad:", grad['fc1']['weight'].tolist())

print("Model parameters have been updated after backpropagation.")
