import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.models as models
import copy
import os
from torch.nn import init
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# 模拟伪分布式训练
num_clients = 4
num_rounds = 100
num_epochs_per_round = 5
eta_c = 0.01  # 客户端超参数，可以根据需要调整
gamma_c = 10  # 客户端超参数，可以根据需要调整
eta_s = 0.05  # 服务器超参数，可以根据需要调整
gamma_s = 100  # 服务器超参数，可以根据需要调整
quantize = False
bit = 8
flag = True  # 是否生成节点数据分布图
alpha = 1.0  # 控制 Non-IID 程度的参数
iid = False  # 是否按IID方式划分数据集
batch_size = 64


class Quantizer:
    def __init__(self, b, *args, **kwargs):
        self.bit = b

    def __str__(self):
        return f"{self.bit}-bit quantization"

    def __call__(self, x):
        with torch.no_grad():
            ma = x.max().item()
            mi = x.min().item()
            if ma == mi:
                return x  # 如果最大值和最小值相等，直接返回原始值
            k = ((1 << self.bit) - 1) / (ma - mi)
            b = -mi * k
            x_qu = torch.round(k * x + b)
            x_qu -= b
            x_qu /= k
            return x_qu

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
seed = 42
set_seed(seed)

# 设置设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 选择数据集
dataset_name = "TinyImageNet"  # 或者 "CIFAR100"

# 数据预处理和增强
if dataset_name == "CIFAR100":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    # 加载 CIFAR-100 数据集
    train_dataset = datasets.CIFAR100(root='/dataset/cifar100', train=True, download=False, transform=transform_train)
    val_dataset = datasets.CIFAR100(root='/dataset/cifar100', train=False, download=False, transform=transform_val)
else:
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载 Tiny ImageNet 数据集
    data_dir = '/dataset/tiny-imagenet-200'  # 修改为 Tiny ImageNet 数据集的路径
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_val)
# 将训练数据集按 Dirichlet 分布划分为多个客户端数据集
def split_dataset(train_dataset, num_clients, alpha, iid):
    if iid:
        return random_split(train_dataset, [len(train_dataset) // num_clients] * num_clients)
    else:
        train_labels = np.array([s[1] for s in train_dataset.samples])
        class_indices = defaultdict(list)
        for idx, label in enumerate(train_labels):
            class_indices[label].append(idx)

        client_indices = [[] for _ in range(num_clients)]
        for indices in class_indices.values():
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(client_idx) < len(train_dataset) / num_clients) for p, client_idx in zip(proportions, client_indices)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            split_indices = np.split(indices, proportions)
            for client_idx, split_idx in zip(client_indices, split_indices):
                client_idx.extend(split_idx)

        return [torch.utils.data.Subset(train_dataset, indices) for indices in client_indices]

client_datasets = split_dataset(train_dataset, num_clients, alpha, iid)
# 将训练数据集拆分为多个客户端数据集
# client_datasets = random_split(train_dataset, [len(train_dataset) // num_clients] * num_clients)
# 绘制数据分布图
if flag:
    labels = [train_dataset.samples[i][1] for i in range(len(train_dataset))]
    plt.figure(figsize=(20, 3))
    plt.hist([np.array(labels)[idx] for idx in [ds.indices for ds in client_datasets]], stacked=True,
             bins=np.arange(min(labels) - 0.5, max(labels) + 1.5, 1),
             label=["Client {}".format(i + 1) for i in range(num_clients)], rwidth=0.5)
    plt.xticks(np.arange(len(set(labels))))
    plt.legend()
    plt.savefig("data_distribution.png")
    plt.show()
# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 加载预训练的 MobileNetV3 Small 模型
global_model = models.mobilenet_v3_small(pretrained=True)

# 获取最后一层的输入特征数
num_ftrs = global_model.classifier[3].in_features

# 替换最后一层，使其适应 200 个类别或 100 个类别
if dataset_name == "CIFAR100":
    global_model.classifier[3] = nn.Linear(num_ftrs, 100)
else:
    global_model.classifier[3] = nn.Linear(num_ftrs, 200)

# 将整个模型转移到 GPU 上
global_model = global_model.to(device)

# 模拟每个客户端的训练
def train_client(model, train_loader, eta_c, gamma_c, num_epochs=1):
    model.train()
    local_epoch_norm_max = 0.0
    local_norm_sum = 0.0
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 反向传播
            model.zero_grad()
            loss.backward()

            # 计算步长 h_{t,j}^i 并更新参数
            state_dict = model.state_dict()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm(p=2)  # 使用 2 范数
                    step_size = min(eta_c, (gamma_c * eta_c) / grad_norm)
                    if step_size != 0.01:
                        print(step_size)
                    state_dict[name] -= step_size * param.grad
            model.load_state_dict(state_dict)
            local_epoch_norm = torch.sqrt(sum(param.grad.norm(p=2) ** 2 for name, param in model.named_parameters()))
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            if local_epoch_norm >= local_epoch_norm_max:
                local_epoch_norm_max = local_epoch_norm
            local_norm_sum += local_epoch_norm
            # print(f'Local Epoch Norm: {local_epoch_norm:.4f}')
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Client Epoch Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 返回更新后的模型
    return model.state_dict(), local_epoch_norm_max, local_norm_sum/(num_epochs_per_round*(len(train_loader.dataset)/batch_size)), epoch_loss

# 聚合客户端模型参数差异并更新全局模型
def aggregate_models(global_model, client_models, bit, quantize):
    global_dict = global_model.state_dict()

    # 初始化参数差异
    param_diffs = {name: torch.zeros_like(param).float() for name, param in global_dict.items()}

    # 计算每个客户端模型参数与全局模型参数的差异
    for client_model in client_models:
        client_dict = client_model.state_dict()
        for name in global_dict.keys():
            param_diffs[name] += global_dict[name] - client_dict[name]

    # 量化参数差异（如果需要）
    if quantize:
        q = Quantizer(bit)
        for name in global_dict.keys():
            param_diffs[name] = q(param_diffs[name])
        print("quantize success")

    param_diffs_norm = torch.sqrt(sum(torch.norm(param_diffs[name] / (num_clients), p=2)**2 for name in param_diffs.keys()))
    global_step_size = min(eta_s, (gamma_s * eta_s) / (param_diffs_norm/(num_clients*num_epochs_per_round)))
    if global_step_size != 0.01:
        print(f'Global Step Size: {global_step_size}')
    # 计算平均参数差异并更新全局模型
    for name in global_dict.keys():
        global_dict[name] = global_dict[name].float() - global_step_size * (param_diffs[name] / (num_clients * eta_c)).float()

    global_model.load_state_dict(global_dict)
    return global_model, param_diffs_norm, global_step_size

# 验证模型
def validate_model(model, val_loader):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = running_corrects.double() / len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    return val_acc


trainloss_file = './trainloss' + '_MobileNetV3_small' + '.txt'
if (os.path.isfile(trainloss_file)):
    os.remove(trainloss_file)
f_trainloss = open(trainloss_file, 'a')


for round in range(num_rounds):
    print(f'Round {round + 1}/{num_rounds}')
    client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
    local_norm_max_all = 0.0
    local_norm_average_all =0.0
    local_loss = 0.0
    for client_id, client_dataset in enumerate(client_datasets):
        train_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        client_model = client_models[client_id]
        updated_state_dict, local_norm_max, local_norm_average, loss = train_client(client_model, train_loader, eta_c, gamma_c, num_epochs=num_epochs_per_round)
        client_model.load_state_dict(updated_state_dict)
        local_norm_max_all += local_norm_max
        local_norm_average_all += local_norm_average
        local_loss += loss
    # 聚合客户端模型参数差异并更新全局模型
    global_model, global_gradient_norm,global_step_size = aggregate_models(global_model, client_models, bit, quantize)

    # 每轮结束后进行验证
    val_acc = validate_model(global_model, DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True))
    print(f'Train Loss: {local_loss/num_clients} Valid Acc: {val_acc:.4f} Global Gradient Norm: {global_gradient_norm} \n'
          f'Local Norm Max: {local_norm_max_all/num_clients} Local Norm Average:{local_norm_average_all/num_clients}')
    f_trainloss.write(str(local_loss/num_clients) + "\t" + (f"{val_acc.item():.4f}") + "\t" + (f"{global_gradient_norm.item()}") + "\t"
                      + (f"{local_norm_max_all.item()/num_clients}")+ "\t" +(f"{local_norm_average_all.item()/num_clients}")+"\t"+(f"{global_step_size.item()}")+ '\n')
    f_trainloss.flush()
