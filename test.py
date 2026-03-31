import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_cifar10_pretrained():
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True, trust_repo=True)
    return model

def train_mnist_model(epochs=1):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def check_cifar10_accuracy(model):
    class NormalizeModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.register_buffer("mean", torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device))
            self.register_buffer("std", torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device))
            
        def forward(self, x):
            x_norm = (x - self.mean) / self.std
            return self.base_model(x_norm)
            
    model = NormalizeModel(model).to(device)
    return model

# ---------------------------------------------------------
# 2. 적대적 공격기법 기초 (FGSM)
# ---------------------------------------------------------
def fgsm_targeted(model, x, target, eps):
    """타겟 FGSM(Targeted FGSM)"""
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    
    grad = x_adv.grad.data
    x_adv_data = x_adv.data - eps * torch.sign(grad)
    x_adv_data = torch.clamp(x_adv_data, 0, 1)
    
    return x_adv_data

def fgsm_untargeted(model, x, label, eps):
    """비타겟 FGSM(Untargeted FGSM)"""
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    
    grad = x_adv.grad.data
    x_adv_data = x_adv.data + eps * torch.sign(grad)
    x_adv_data = torch.clamp(x_adv_data, 0, 1)
    
    return x_adv_data

def evaluate_fgsm(model, device, test_loader, attack_name, epsilons):
    print(f"\n{attack_name} 평가 중...")
    model.eval()
    for eps in epsilons:
        correct = 0
        total = 0
        success = 0
        is_targeted = "Targeted" in attack_name
        
        for images, labels in test_loader:
            if total >= 200: # 검증용으로 빠르게 200개만 측정
                break
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                clean_outputs = model(images)
                _, clean_preds = torch.max(clean_outputs, 1)
                
            mask = (clean_preds == labels)
            if not mask.any(): continue
            images, labels, clean_preds = images[mask], labels[mask], clean_preds[mask]
            
            targets = (labels + 1) % 10 if is_targeted else None
            
            if is_targeted:
                adv_images = fgsm_targeted(model, images, targets, eps)
            else:
                adv_images = fgsm_untargeted(model, images, labels, eps)
                
            with torch.no_grad():
                adv_outputs = model(adv_images)
                _, adv_preds = torch.max(adv_outputs, 1)
            
            total += len(labels)
            if is_targeted:
                success += (adv_preds == targets).sum().item()
            else:
                success += (adv_preds != labels).sum().item()
                
        success_rate = (success / total) * 100 if total > 0 else 0
        print(f"Epsilon: {eps:.2f} \t 공격 성공률: {success_rate:.2f}%")

if __name__ == '__main__':
    print("개발 4단계: 기초 적대적 공격기법 FGSM 구현 완료!")
    mnist_model = train_mnist_model(epochs=1)
    
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    epsilons = [0.05, 0.1, 0.3]
    evaluate_fgsm(mnist_model, device, test_loader, "Untargeted FGSM", epsilons)
