import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm
from PIL import Image

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"사용 중인 디바이스: {device}")

# 난수 시드 고정
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

# ---------------------------------------------------------
# 1. 모델 정의
# ---------------------------------------------------------

class MNIST_CNN(nn.Module):
    """
    MNIST 분류를 위한 간단한 CNN 아키텍처
    1x28x28 흑백 이미지를 입력으로 받습니다.
    """
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 28x28 -> pool -> 14x14 -> pool -> 7x7
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
    """
    CIFAR-10용으로 사전 학습된 ResNet-20 모델을 다운로드하고 반환합니다.
    출처(인용): chenyaofo/pytorch-cifar-models (assignment 가이드라인에 따른 오픈소스 모델 사용 허용)
    """
    print("torch.hub를 통해 사전 학습된 CIFAR-10 resnet20 모델을 불러옵니다...")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True, trust_repo=True)
    return model

# ---------------------------------------------------------
# 2. 적대적 공격(Adversarial Attacks) 구현
# ---------------------------------------------------------

def fgsm_targeted(model, x, target, eps):
    """
    타겟 FGSM(Targeted FGSM) 공격
    model : 신경망 모델
    x : 입력 이미지 텐서 (requires_grad가 필요함)
    target : 원하는(잘못된) 타겟 클래스 레이블
    eps : 노이즈 정도(perturbation magnitude)
    return : 적대적 이미지(adversarial image) x_adv
    """
    x_adv = x.clone().detach().requires_grad_(True)
    
    output = model(x_adv)
    loss = F.cross_entropy(output, target)
    
    model.zero_grad()
    loss.backward()
    
    # 타겟에 대한 loss를 최소화하도록 이동: x = x - eps * sign(grad)
    grad = x_adv.grad.data
    x_adv_data = x_adv.data - eps * torch.sign(grad)
    x_adv_data = torch.clamp(x_adv_data, 0, 1) # 이미지 범위를 [0,1]로 제한
    
    return x_adv_data

def fgsm_untargeted(model, x, label, eps):
    """
    비타겟 FGSM(Untargeted FGSM) 공격
    """
    x_adv = x.clone().detach().requires_grad_(True)
    
    output = model(x_adv)
    loss = F.cross_entropy(output, label)
    
    model.zero_grad()
    loss.backward()
    
    # 올바른 정답 레이블에 대한 loss를 최대화하도록 이동: x = x + eps * sign(grad)
    grad = x_adv.grad.data
    x_adv_data = x_adv.data + eps * torch.sign(grad)
    x_adv_data = torch.clamp(x_adv_data, 0, 1) # 이미지 범위를 [0,1]로 제한
    
    return x_adv_data

def pgd_targeted(model, x, target, k, eps, eps_step):
    """
    타겟 PGD(Targeted Projected Gradient Descent) 공격
    """
    x_adv = x.clone().detach()
    
    for _ in range(k):
        x_adv.requires_grad = True
        
        output = model(x_adv)
        loss = F.cross_entropy(output, target)
        
        model.zero_grad()
        loss.backward()
        
        grad = x_adv.grad.data
        
        # 타겟에 대한 loss 최소화: 기울기 빼기
        x_adv = x_adv.data - eps_step * torch.sign(grad)
        
        # 원래 이미지의 eps 반경 안으로 투영(projection)
        eta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + eta, 0, 1).detach()
        
    return x_adv

def pgd_untargeted(model, x, label, k, eps, eps_step):
    """
    비타겟 PGD(Untargeted Projected Gradient Descent) 공격
    """
    x_adv = x.clone().detach()
    
    for _ in range(k):
        x_adv.requires_grad = True
        
        output = model(x_adv)
        loss = F.cross_entropy(output, label)
        
        model.zero_grad()
        loss.backward()
        
        grad = x_adv.grad.data
        
        # 정답에 대한 loss 최대화: 기울기 더하기
        x_adv = x_adv.data + eps_step * torch.sign(grad)
        
        # 원래 이미지의 eps 반경 안으로 투영(projection)
        eta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + eta, 0, 1).detach()
        
    return x_adv

# ---------------------------------------------------------
# 3. 학습 및 모델 평가 파이프라인
# ---------------------------------------------------------

def train_mnist_model(epochs=3):
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
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
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
    
    # 입력 범위를 [0, 1]로 유지하면서 chenyaofo 모델에 필요한 정규화를 적용하는 래퍼 모델 (권장)
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

# ---------------------------------------------------------
# 4. 공격 실행 및 결과 저장 파이프라인
# ---------------------------------------------------------

def evaluate_attack(model, device, test_loader, attack_name, dataset_name, epsilons, k=40, eps_step=0.01):
    print(f"\n{dataset_name} 에 대한 {attack_name} 평가 중...")
    model.eval()
    
    results = {}
    
    # 가장 작은 epsilon(가장 은밀한 공격)에서 시각화 이미지를 저장합니다.
    # → 사람 눈에 거의 인지 불가능한 수준의 섭동임을 보여주기 위함
    min_eps = min(epsilons)
    
    for eps in epsilons:
        correct = 0
        total = 0
        success = 0
        
        saved_images = 0
        is_targeted = "Targeted" in attack_name
        
        # 평가 속도를 최적화하기 위해, 공격별로 최소 100 샘플만 평가해도 되지만 500 샘플을 사용합니다.
        num_eval_samples = 500
        
        for images, labels in test_loader:
            # 500개 샘플 평가 후 종료하되, min_eps에서 시각화할 이미지를 5장 다 못 모았다면 추가 탐색
            if total >= num_eval_samples and not (eps == min_eps and saved_images < 5):
                break
                
            images, labels = images.to(device), labels.to(device)
            
            # 공격 전 Clean 상태 예측
            with torch.no_grad():
                clean_outputs = model(images)
                _, clean_preds = torch.max(clean_outputs, 1)
                
            # 분류 모델이 먼저 정답을 맞힌 샘플에 대해서만 공격 성공 여부 검사 
            mask = (clean_preds == labels)
            if not mask.any():
                continue
                
            images, labels, clean_preds = images[mask], labels[mask], clean_preds[mask]
            
            # 타겟 라벨 설정 (클래스를 1씩 변경, 예: 0->1, 9->0)
            targets = (labels + 1) % 10 if is_targeted else None
            
            # 적대적 예제(Adversary) 생성
            if attack_name == "Targeted FGSM":
                adv_images = fgsm_targeted(model, images, targets, eps)
            elif attack_name == "Untargeted FGSM":
                adv_images = fgsm_untargeted(model, images, labels, eps)
            elif attack_name == "Targeted PGD":
                adv_images = pgd_targeted(model, images, targets, k, eps, eps_step)
            elif attack_name == "Untargeted PGD":
                adv_images = pgd_untargeted(model, images, labels, k, eps, eps_step)
            
            # 생성된 적대적 예제로 모델 예측
            with torch.no_grad():
                adv_outputs = model(adv_images)
                _, adv_preds = torch.max(adv_outputs, 1)
            
            total += len(labels)
            
            if is_targeted:
                # 공격 성공: 모델 예측이 우리가 설정한 타겟 클래스와 일치해야 함
                success += (adv_preds == targets).sum().item()
            else:
                # 공격 성공: 모델 예측이 실제 정답과 달라져야 함
                success += (adv_preds != labels).sum().item()
                
            # 가장 작은 eps(가장 은밀한 공격)에서 이미지를 5장 저장합니다.
            # 사람이 육안으로 거의 구별 불가능한 수준의 섭동만 시각화합니다.
            if eps == min_eps and saved_images < 5:
                for i in range(len(labels)):
                    if saved_images >= 5:
                        break
                        
                    # 공격에 성공한 샘플의 이미지만 저장
                    attack_succeeded = (adv_preds[i] == targets[i]) if is_targeted else (adv_preds[i] != labels[i])
                    if attack_succeeded:
                        # CPU의 numpy 데이터로 변환
                        orig_img = images[i].cpu().squeeze().numpy()
                        adv_img = adv_images[i].cpu().squeeze().numpy()
                        clean_pred_cls = clean_preds[i].item()
                        adv_pred_cls = adv_preds[i].item()
                        true_cls = labels[i].item()
                        tgt_cls = targets[i].item() if targets is not None else -1
                        
                        # 화면 시각화 및 저장
                        save_visualization(orig_img, adv_img, clean_pred_cls, adv_pred_cls, true_cls, tgt_cls,
                                           attack_name, dataset_name, eps, saved_images)
                        saved_images += 1
                        
        success_rate = (success / total) * 100 if total > 0 else 0
        results[eps] = success_rate
        print(f"Epsilon: {eps:.2f} \t 공격 성공률: {success_rate:.2f}%")
        
    return results

def upsample_img(img_np, scale=4):
    """
    PIL bicubic 업샘플링으로 이미지를 scale배 확대합니다.
    CIFAR-10처럼 저해상도(32x32) 이미지의 시각화 품질을 높이기 위해 사용합니다.
    """
    # float32 -> uint8 변환 후 PIL로 업샘플링
    img_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    new_size = (pil_img.width * scale, pil_img.height * scale)
    pil_img = pil_img.resize(new_size, Image.BICUBIC)
    return np.array(pil_img).astype(np.float32) / 255.0

def save_visualization(orig_img, adv_img, clean_pred, adv_pred, true_cls, tgt_cls, attack_name, dataset_name, eps, idx):
    # 공격 방법과 타겟 여부 추출
    method = 'FGSM' if 'FGSM' in attack_name else 'PGD'
    target_type = 'Targeted' if 'Targeted' in attack_name else 'Untargeted'
    
    # 결과를 저장할 디렉토리 경로: results/{dataset}/{method}/{target_type}
    save_dir = os.path.join('results', dataset_name, method, target_type)
    os.makedirs(save_dir, exist_ok=True)
        
    # 이미지가 (Channels, Height, Width) 이면 시각화를 위해 (H, W, C)로 변경
    if len(orig_img.shape) == 3 and orig_img.shape[0] == 3:
        orig_img = np.transpose(orig_img, (1, 2, 0))
        adv_img = np.transpose(adv_img, (1, 2, 0))
        cmap = None
        is_color = True
    else:
        cmap = 'gray'
        is_color = False
    
    # CIFAR-10(32x32)의 화질이 지나치게 낮아 보이는 문제를 PIL bicubic 업샘플링으로 보완
    if is_color:
        orig_img = upsample_img(orig_img, scale=4)   # 32x32 -> 128x128
        adv_img  = upsample_img(adv_img,  scale=4)
    
    # 노이즈(perturbation)를 시각적으로 확대하여 표시 (x15)
    # 실제 섭동은 매우 미세하기 때문에 크게 증폭해야 눈에 보임
    perturbation = np.clip(np.abs(adv_img - orig_img) * 15, 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    
    # CIFAR-10 클래스명 매핑 (숫자 -> 문자열 변환)
    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    if dataset_name == 'CIFAR10':
        clean_pred_str = cifar_classes[clean_pred]
        adv_pred_str   = cifar_classes[adv_pred]
        true_cls_str   = cifar_classes[true_cls]
        tgt_cls_str    = cifar_classes[tgt_cls] if tgt_cls != -1 else "-1"
    else:
        clean_pred_str, adv_pred_str = str(clean_pred), str(adv_pred)
        true_cls_str, tgt_cls_str = str(true_cls), str(tgt_cls)
    
    axes[0].imshow(orig_img, cmap=cmap, interpolation='bilinear')
    axes[0].set_title(f"Original\nPred: {clean_pred_str} (True: {true_cls_str})")
    axes[0].axis('off')
    
    if "Targeted" in attack_name:
        title_adv = f"Adversarial (ε={eps})\nPred: {adv_pred_str} (Target: {tgt_cls_str})"
    else:
        title_adv = f"Adversarial (ε={eps})\nPred: {adv_pred_str}"
        
    axes[1].imshow(adv_img, cmap=cmap, interpolation='bilinear')
    axes[1].set_title(title_adv)
    axes[1].axis('off')
    
    axes[2].imshow(perturbation, cmap=cmap, interpolation='bilinear')
    axes[2].set_title(f"Perturbation\n(Magnified ×15)")
    axes[2].axis('off')
    
    plt.tight_layout()
    filename = os.path.join(save_dir, f"eps{eps:.2f}_sample{idx}.png")
    plt.savefig(filename, dpi=150)
    plt.close()

def main():
    print("====== Adversarial Attacks 평가 시작 ======")
    
    # MNIST 모델 학습
    mnist_model = train_mnist_model(epochs=3)
    
    # CIFAR-10 모델 불러오기 및 평가
    cifar_base_model = get_cifar10_pretrained()
    cifar10_model = check_cifar10_accuracy(cifar_base_model)
    
    # 평가용 데이터 로더 정의 (배치 사이즈 32)
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)
    mnist_loader = DataLoader(mnist_test, batch_size=32, shuffle=True)
    
    cifar10_transform = transforms.Compose([transforms.ToTensor()])
    cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar10_transform)
    cifar10_loader = DataLoader(cifar10_test, batch_size=32, shuffle=True)
    
    # 과제에서 권장된 Epsilon 설정 및 공격 메서드들
    epsilons = [0.05, 0.1, 0.2, 0.3]
    attacks = ["Targeted FGSM", "Untargeted FGSM", "Targeted PGD", "Untargeted PGD"]
    
    # PGD 의 하이퍼파라미터
    mnist_k = 40
    mnist_eps_step = 0.01
    
    cifar_k = 10
    cifar_eps_step = 0.02
    
    results_summary = {}
    
    # 1. MNIST 결과 평가
    print("\n" + "="*40)
    print("MNIST 데이터셋 공격 평가 진행")
    print("="*40)
    results_summary['MNIST'] = {}
    for attack in attacks:
        results = evaluate_attack(mnist_model, device, mnist_loader, attack, "MNIST", epsilons, mnist_k, mnist_eps_step)
        results_summary['MNIST'][attack] = results
        
    # 2. CIFAR-10 결과 평가
    print("\n" + "="*40)
    print("CIFAR-10 데이터셋 공격 평가 진행")
    print("="*40)
    results_summary['CIFAR10'] = {}
    for attack in attacks:
        results = evaluate_attack(cifar10_model, device, cifar10_loader, attack, "CIFAR10", epsilons, cifar_k, cifar_eps_step)
        results_summary['CIFAR10'][attack] = results
        
    print("\n====== 전체 평가 완료 ======")
    print("시각화 결과 이미지들은 'results' 폴더에 저장되었습니다.")
    
    # 공격 성공률 요약 화면 출력
    print("\n--- 공격 성공률 요약(Markdown Table) ---")
    print("| 데이터셋 | 공격법 | $\\epsilon=0.05$ | $\\epsilon=0.10$ | $\\epsilon=0.20$ | $\\epsilon=0.30$ |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for dataset in ['MNIST', 'CIFAR10']:
        for attack in attacks:
            row_res = results_summary[dataset][attack]
            print(f"| {dataset} | {attack} | {row_res.get(0.05,0):.1f}% | {row_res.get(0.1,0):.1f}% | {row_res.get(0.2,0):.1f}% | {row_res.get(0.3,0):.1f}% |")

if __name__ == '__main__':
    main()
