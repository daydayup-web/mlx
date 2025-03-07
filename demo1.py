import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

class SimpleNet(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim1, output_dim2):
        super(SimpleNet, self).__init__()
        self.mul1 = nn.Linear(input_dim1, output_dim1, False)
        self.mul2 = nn.Linear(input_dim2, output_dim2, False)
        print("Weight of mul1 shape:", self.mul1.weight.shape)
        print("Weight of mul2 shape:", self.mul2.weight.shape)
        # print(self.mul1.weight.tolist())

    def __call__(self, x):
        out1 = self.mul1(x)
        out2 = self.mul2(x)
        sin1 = mx.sin(out1)
        cos1 = mx.cos(out1)
        sin2 = mx.sin(out2)
        cos2 = mx.cos(out2)
        concat_out1 = mx.concatenate([sin1, cos1], 1)
        concat_out2 = mx.concatenate([sin2, cos2], 1)
        out = mx.concatenate([concat_out1, concat_out2], 1)
        return out

input_size1 = 1
input_size2 = 1
output_size1 = 160
output_size2 = 64
epoch = 5

model = SimpleNet(input_size1, input_size2, output_size1, output_size2)

batch_size = 1
x = mx.random.uniform(-1, 1, shape=(batch_size, input_size1))
# print("Input shape:", x.shape)
# print(x.tolist())

# 标签
eps = 1e-2 * mx.random.normal((batch_size, 448))
# print("eps shape:", eps.shape)
target = eps
# print("Target shape:", target.shape)
# print(target.tolist())

# 定义损失函数
def loss(models, inputs, targets):
    pred = models(inputs)
    # print("Output shape:", pred.shape)
    # print(pred.tolist())
    ce = nn.losses.mse_loss(pred, targets)
    return ce.mean()

# 定义优化器
learning_rate = 0.01
optimizer = optim.Adam(learning_rate)

# 前向传播&反向传播
loss_value_and_grad = nn.value_and_grad(model, loss)
for i in range(epoch):
    print(f"第 {i + 1} epoch")
    loss, grad = loss_value_and_grad(model, x, target)
    print("Loss:", loss)
    print("Grad shape:", grad['mul1']['weight'].shape)
    print("Grad:", grad['mul1']['weight'].tolist())
    print("Grad shape:", grad['mul2']['weight'].shape)
    print("Grad:", grad['mul2']['weight'].tolist())
    optimizer.update(model, grad)
    mx.eval(model.parameters(), optimizer.state, loss)

print("Model parameters have been updated after backpropagation.")
