# Trustworthy AI 적대적 공격 프로젝트

## 개요
- 본 레포지토리는 **MNIST**와 **CIFAR‑10** 데이터셋에 대해 **FGSM** 및 **PGD** 적대적 공격을 구현합니다.
- 공격은 **Targeted**(목표 지정)와 **Untargeted**(목표 미지정) 두 종류를 모두 지원합니다.
- 결과 시각화 파일은 `results/{데이터셋}/{공격방법}/{Targeted|Untargeted}/` 형태의 계층적 디렉터리 구조에 저장됩니다.
- Apple Silicon(M1/M2) GPU를 활용하기 위해 **MPS** 지원을 추가했으며, CUDA와 CPU에도 자동으로 fallback 합니다.
- CIFAR‑10 이미지의 저해상도(32×32) 문제를 해결하기 위해 **PIL Bicubic 업샘플링**을 적용해 시각화 품질을 향상시켰습니다.

## 요구 사항 (requirements.txt)
```bash
pip install -r requirements.txt
```
`requirements.txt` 에 포함된 패키지:
```
torch
torchvision
numpy
matplotlib
tqdm
pillow
```

## 환경 설정 및 실행 방법
1. **레포지토리 복제**
   ```bash
   git clone https://github.com/ashmin100/trustworthy-ai-adversarial-attacks.git
   cd trustworthy-ai-adversarial-attacks
   ```
2. **가상 환경 생성 및 활성화** (권장)
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   # Windows: venv\Scripts\activate
   ```
3. **필요 패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```
4. **파이프라인 실행**
   - 메인 스크립트는 `test.py`입니다.
   ```bash
   python test.py
   ```
   실행 시 수행되는 작업:
   - MNIST 모델을 3 epoch 동안 학습하고 정확도(≥95%)를 확인합니다.
   - 사전 학습된 CIFAR‑10 ResNet‑20 모델을 `torch.hub` 로 다운로드 받아 로드합니다 (정확도 ≥80%).
   - 각 공격 방법(FGSM, PGD)·타입(Targeted, Untargeted)·ε 값(0.05, 0.10, 0.20, 0.30)에 대해 500개의 샘플을 평가합니다.
   - 가장 작은 ε(가장 은밀한 공격)에서 성공한 샘플을 5장씩 시각화하고, `results/` 디렉터리에 저장합니다.
   - 콘솔에 마크다운 형식의 성공률 표를 출력합니다.

## 결과 구조
- **시각화 파일**은 다음과 같은 경로에 저장됩니다.
  ```
  results/<DATASET>/<METHOD>/<Targeted|Untargeted>/eps<ε>_sample<IDX>.png
  ```
  예시: `results/MNIST/FGSM/Targeted/eps0.05_sample0.png`
- **요약 표**는 실행 종료 시 콘솔에 마크다운 테이블 형태로 출력됩니다.

# 주요 구현 상세 (Adversarial Attack Implementation)

## 1. FGSM (Fast Gradient Sign Method)

FGSM은 입력 이미지에 대해 손실 함수의 gradient 방향으로 perturbation을 추가하는 단일 단계(one-step) 공격 기법이다.

수식:
$$
x_{adv} = x + \epsilon \cdot \mathrm{sign}(\nabla_x J(\theta, x, y))
$$

구현에서는 gradient의 sign만 사용하여 계산을 단순화하고, epsilon을 통해 perturbation 크기를 제어한다.

또한 Targeted / Untargeted 공격을 다음과 같이 처리한다:

```python
if is_targeted:
    grad = -grad
else:
    grad = grad

x_adv = x + epsilon * grad.sign()
```

- Untargeted: 정답에서 멀어지도록 (+gradient)
- Targeted: 특정 클래스에 가깝게 (-gradient)

---

## 2. PGD (Projected Gradient Descent)

PGD는 FGSM을 여러 번 반복 적용하는 iterative 공격 기법으로, 더 강력한 adversarial example을 생성한다.

수식:
$$
x^{t+1} = \Pi_{B_\epsilon(x)} \left( x^t + \alpha \cdot \mathrm{sign}(\nabla_x J(\theta, x^t, y)) \right)
$$

구현에서는 다음과 같이 반복적으로 gradient를 적용한다:

```python
for _ in range(num_steps):
    x_adv.requires_grad = True
    outputs = model(x_adv)
    loss = criterion(outputs, target)

    if is_targeted:
        loss = -loss

    loss.backward()
    grad = x_adv.grad.data

    x_adv = x_adv + alpha * grad.sign()

    # projection (L∞ constraint)
    x_adv = torch.clamp(x_adv, x - eps, x + eps)
    x_adv = torch.clamp(x_adv, 0, 1)
```

PGD는 FGSM보다 더 정교하고 강력한 공격을 수행할 수 있다.

---

## 3. Targeted vs Untargeted Attack

| 타입 | 목적 |
|------|------|
| Untargeted | 정답이 아니게 만들기 |
| Targeted | 특정 클래스로 오분류 유도 |

구현:

```python
is_targeted = (attack_name == "Targeted FGSM" or attack_name == "Targeted PGD")
```

- 정확한 문자열 매칭으로 공격 타입을 구분하여 실험 재현성을 확보한다.

Gradient 방향:

- Untargeted → +gradient  
- Targeted → -gradient  

---

## 4. Perturbation 제한 (L∞ Constraint)

모든 공격은 다음 조건을 만족하도록 제한된다:

$$
||x_{adv} - x||_\infty \le \epsilon
$$

구현:

```python
x_adv = torch.clamp(x_adv, x - eps, x + eps)
```

- 각 픽셀의 최대 변화량을 제한하여 사람이 인지하기 어려운 perturbation을 유지한다.

---

## 5. 결과 저장 구조 자동화

```text
results/
 └── CIFAR10/
      └── FGSM/
           └── Targeted/
```

구현:

```python
if attack_name == "Targeted FGSM" or attack_name == "Untargeted FGSM":
    method = "FGSM"
else:
    method = "PGD"

if attack_name == "Targeted FGSM" or attack_name == "Targeted PGD":
    target_type = "Targeted"
else:
    target_type = "Untargeted"

save_dir = os.path.join('results', dataset_name, method, target_type)
os.makedirs(save_dir, exist_ok=True)
```

---

## 6. 시각화 (Visualization)

구성:
- Original Image
- Adversarial Image
- Perturbation (×15 scaling)

구현:

```python
noise_vis = perturbation * 15
```

업샘플링:

```python
def upsample_img(img_np, scale=4):
    img_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    new_size = (pil_img.width * scale, pil_img.height * scale)
    pil_img = pil_img.resize(new_size, Image.BICUBIC)
    return np.array(pil_img).astype(np.float32) / 255.0
```

- CIFAR-10 (32×32) 이미지를 128×128로 확대하여 시각적 해석을 용이하게 한다.

---

## 요약

- FGSM: 빠른 1-step 공격
- PGD: 반복 기반 강력한 공격
- Targeted / Untargeted 명확 분리
- L∞ constraint 기반 perturbation 제어
- 시각화 및 저장 구조까지 포함한 end-to-end 공격 분석 파이프라인 구현

## 참고 사항
- **Targeted 공격 성공 기준**: 모델 예측이 지정한 목표 클래스와 일치해야 합니다.
- **Untargeted 공격 성공 기준**: 모델 예측이 원래 정답과 달라야 합니다.
- `results/` 폴더는 자동으로 생성되며, 기존 파일이 있으면 덮어쓰지 않고 새 폴더에 저장됩니다.

## 라이선스
학술 목적 사용을 전제로 합니다. 자유롭게 포크, 수정, 재배포 가능하지만 상업적 이용 시 별도 허가를 받아 주세요.
