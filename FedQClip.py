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
eta_c = 0.05  
gamma_c = 10  
eta_s = 0.05 
gamma_s = 1000000  
quantize = True
bit = 4
flag = True  
alpha = 1.0  
iid = False  
batch_size = 64
model_name = "MobileNetV3"  # or "ResNet18"

dataset_name = "TinyImagenet"  # or "CIFAR100" or "CIFAR10"



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
                return x  
            k = ((1 << self.bit) - 1) / (ma - mi)
            b = -mi * k
            x_qu = torch.round(k * x + b)
            x_qu -= b
            x_qu /= k
            return x_qu


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_seed(seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




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


    data_dir = '/dataset/tiny-imagenet-200' 
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_val)


def split_dataset(train_dataset, num_clients, alpha, iid):
    if iid:
        return random_split(train_dataset, [len(train_dataset) // num_clients] * num_clients)
    else:
        if hasattr(train_dataset, 'targets'):
            train_labels = train_dataset.targets
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


if flag:
    if hasattr(train_dataset, 'targets'):
        labels = train_dataset.targets
    else:
        labels = [train_dataset.samples[i][1] for i in range(len(train_dataset))]
    plt.figure(figsize=(20, 3))
    plt.hist([np.array(labels)[idx] for idx in [ds.indices for ds in client_datasets]], stacked=True,
             bins=np.arange(min(labels) - 0.5, max(labels) + 1.5, 1),
             label=["Client {}".format(i + 1) for i in range(num_clients)], rwidth=0.5)
    plt.xticks(np.arange(len(set(labels))))
    plt.legend()
    plt.savefig("data_distribution.png")
    plt.show()


criterion = nn.CrossEntropyLoss()


def get_model(model_name, dataset_name, device):
    if model_name == "MobileNetV3":
        model = models.mobilenet_v3_small(pretrained=True)
        num_ftrs = model.classifier[3].in_features
        if dataset_name == "CIFAR100":
            model.classifier[3] = nn.Linear(num_ftrs, 100)
        else:
            model.classifier[3] = nn.Linear(num_ftrs, 200)
    elif model_name == "ResNet18":
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        if dataset_name == "CIFAR100":
            model.fc = nn.Linear(num_ftrs, 100)
        else:
            model.fc = nn.Linear(num_ftrs, 200)
    model = model.to(device)
    return model


global_model = get_model(model_name, dataset_name, device)


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

         
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

          
            model.zero_grad()
            loss.backward()

            
            state_dict = model.state_dict()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm(p=2)  
                    step_size = min(eta_c, (gamma_c * eta_c) / grad_norm)
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

    
    return model.state_dict(), local_epoch_norm_max, local_norm_sum/(num_epochs_per_round*(len(train_loader.dataset)/batch_size)), epoch_loss

def aggregate_models(global_model, client_models, bit, quantize):
    global_dict = global_model.state_dict()


    param_diffs = {name: torch.zeros_like(param).float() for name, param in global_dict.items()}

   
    for client_model in client_models:
        client_dict = client_model.state_dict()
        for name in global_dict.keys():
            param_diffs[name] += global_dict[name] - client_dict[name]

    
    if quantize:
        q = Quantizer(bit)
        for name in global_dict.keys():
            param_diffs[name] = q(param_diffs[name])
        print("quantize success")

    param_diffs_norm = torch.sqrt(sum(torch.norm(param_diffs[name] / (num_clients), p=2)**2 for name in param_diffs.keys()))
    global_step_size = min(eta_s, (gamma_s * eta_s) / (param_diffs_norm/(num_clients*num_epochs_per_round)))
   
    for name in global_dict.keys():
        global_dict[name] = global_dict[name].float() - global_step_size * (param_diffs[name] / (num_clients * eta_c)).float()

    global_model.load_state_dict(global_dict)
    return global_model, param_diffs_norm, global_step_size


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


trainloss_file = './trainloss' + '_'+model_name+'.txt'
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

    global_model, global_gradient_norm,global_step_size = aggregate_models(global_model, client_models, bit, quantize)


    val_acc = validate_model(global_model, DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True))
    print(f'Train Loss: {local_loss/num_clients} Valid Acc: {val_acc:.4f} Global Gradient Norm: {global_gradient_norm} \n'
          f'Local Norm Max: {local_norm_max_all/num_clients} Local Norm Average:{local_norm_average_all/num_clients}')
    f_trainloss.write(str(local_loss/num_clients) + "\t" + (f"{val_acc.item():.4f}") + "\t" + (f"{global_gradient_norm.item()}") + "\t"
                      + (f"{local_norm_max_all.item()/num_clients}")+ "\t" +(f"{local_norm_average_all.item()/num_clients}")+"\t"+(f"{global_step_size}")+ '\n')
    f_trainloss.flush()
