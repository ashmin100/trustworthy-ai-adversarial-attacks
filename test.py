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
print(f"사용 중인 디바이스: {device}")

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
    print("torch.hub를 통해 사전 학습된 CIFAR-10 resnet20 모델을 불러옵니다...")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True, trust_repo=True)
    return model

def train_mnist_model(epochs=1):
    print("\n--- MNIST 모델 학습 시작 ---")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    # 깨끗한(Clean) 데이터셋 정확도 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    print(f"MNIST Clean 테스트 정확도: {acc:.2f}% (목표 달성 기준: >= 95%)")
    return model

def check_cifar10_accuracy(model):
    print("\n--- 사전 학습된 CIFAR-10 모델 정확도 검증 ---")
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
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    print(f"CIFAR-10 Clean 테스트 정확도: {acc:.2f}% (목표 달성 기준: >= 80%)")
    return model

if __name__ == '__main__':
    print("개발 3단계: 두 데이터셋의 베이스라인 모델 확보 완료!")
    mnist_model = train_mnist_model(epochs=1)
    
    cifar_base_model = get_cifar10_pretrained()
    cifar10_model = check_cifar10_accuracy(cifar_base_model)
