import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from model.mnist import MNISTModel
# from main_mnist_00_train import Net


# 定义网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        # self.scale_x = (2.8215 - (-0.4242)) / 255
        self.scale_x = 2.8215 / 127
        self.x_max = 2.8215
        self.x_min = -0.4242

        self.x_abs_max = 2.8215

        # self.fc1_scale_wt = 0.1242 / 127

        # self.fc1_scale_b = self.scale_x * self.fc1_scale_wt

        self.out_fc1_scale = 9.0297 / 127

    def quant(self):
        fc1_scale_wt = self.fc1.weight.abs().max() / 127

        self.state_dict()['fc1.weight'].copy_(torch.round(torch.clamp(self.fc1.weight/fc1_scale_wt, min=-127, max=127)))

        # self.load_state_dict({'fc1.weight': torch.round(torch.clamp(self.fc1.weight/fc1_scale_wt, min=-127, max=127))})
        # self.fc1.weight = 
        # self.fc1.bias = torch.round(torch.clamp(self.fc1.bias / (self.scale_x*fc1_scale_wt), min=-127, max=127))


    def forward(self, x):
        x = x.view(-1, 784)

        # x = torch.round(torch.clamp(x, self.x_min, self.x_max)/self.scale_x)
        # x = torch.round(torch.clamp(x, -self.x_abs_max, self.x_abs_max)/self.scale_x)

        # x = torch.round(self.fc1(x))

        x = self.fc1(x)

        # x * self.out_fc1_scale

        x = torch.relu(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":

    ckpt = 'ckpt/mnist.pth'
    os.makedirs('/'.join(ckpt.split('/')[:-1]), exist_ok=True)

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    # 定义数据预处理的转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化图像数据
    ])

    # 加载训练集和测试集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


    ########## eval ###########
    model = Net()
    state_dict = torch.load(ckpt)
    model.load_state_dict(state_dict)

    model.quant()

    model.to(device)

    # 在测试集上评估模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')

    ######### quant ###########

    # # 创建模型实例
    # model = Net()

    #### 获取静态图 ####

    # 创建输入张量
    input_tensor = torch.randn(1, 784)

    # 使用跟踪器跟踪前向传播过程
    traced_model = torch.jit.trace(model, input_tensor)

    # 保存跟踪模块
    traced_model_path = "quant/mnist_graph.pt"
    os.makedirs('/'.join(traced_model_path.split('/')[:-1]), exist_ok=True)
    traced_model.save(traced_model_path)
    print('Saved:', traced_model_path)

    # 遍历跟踪模块中的各个层
    for name, module in traced_model.named_modules():
        # if isinstance(module, torch.jit._trace.TracedModule)
        # if isinstance(module, nn.Linear):
        #     # 如果是线性层（nn.Linear），进行相应的操作
        #     print(f"Layer name: {name}")
        #     print(f"Layer module: {module}")
        if (module.original_name == 'Linear'):
            print(f"Layer name: {name}")
            print(f"Layer module: {module}")
            print('w_min:', module.state_dict()['weight'].min(), 'w_max:', module.state_dict()['weight'].max(),
                  'b_min:', module.state_dict()['bias'].min(), 'b_max:', module.state_dict()['bias'].max())
