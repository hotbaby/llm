# encoding: utf8


import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary


class Model(nn.Module):
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


batch_size = 32
input_size = 100
hidden_size = 1000
output_size = 10

# 定义模型、损失函数和优化器
model = Model(input_size, hidden_size, output_size)
model = model.cuda()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = optim.SGD(model.parameters(), lr=1e-3,)


def train(epoch_num):
    # 反向传播，更新参数
    for i in range(epoch_num):
        x = torch.randn(batch_size, input_size).cuda()
        label = torch.randn(batch_size, output_size).cuda()

        y_pred = model(x)
        loss = loss_fn(y_pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch {i} loss: {loss.item()}")


# summary(model, (input_size,), batch_size=32, device="cuda")

def print_cuda_memory():
    print(f"cuda memory_allocated: {torch.cuda.memory_allocated()//1024} KB")
    print(f"cuda max_memory_allocated: {torch.cuda.max_memory_allocated()//1024} KB")
    print(f"cuda max_memory_reserved: {torch.cuda.max_memory_reserved()//1024} KB")


def print_model_params(model: nn.Module):
    for name, value in list(model.named_parameters()):
        print(name, value.size(), value.grad.size(), value.device)


def print_optimizer_parameters():
    state_dict = optimizer.state_dict()
    param_groups = state_dict["param_groups"]
    state = state_dict["state"]

    for i, p in enumerate(param_groups[0]["params"]):
        print(f"optimizer parameter {i}, dtype: {p.dtype}, shape {p.size()}")

    print(state.keys())


def recursive_print_params(state, prefix_key=""):
    if isinstance(state, torch.Tensor):
        print(hex(id(state)), prefix_key, state.size())
    elif isinstance(state, dict):
        for k, v in state.items():
            key = ".".join([prefix_key, str(k)]) if prefix_key else str(k)
            recursive_print_params(v, key)
    elif isinstance(state, list):
        for i, v in enumerate(state):
            key = ".".join([prefix_key, str(i)]) if prefix_key else str(i)
            recursive_print_params(v, key)


train(24)
# print_optimizer_parameters()
state = optimizer.state_dict()["state"]
recursive_print_params(state)
