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

## 주요 구현 상세
### 1. 디바이스 자동 선택
```python
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
```
- CUDA → MPS → CPU 순서로 자동 감지합니다.

### 2. 정확한 공격 타입 판별
```python
# 기존: is_targeted = "Targeted" in attack_name  (오류 발생 가능)
# 수정 후:
is_targeted = (attack_name == "Targeted FGSM" or attack_name == "Targeted PGD")
```
- 문자열 포함이 아닌 **정확히 일치**하는 경우에만 Targeted 로 판단합니다.

### 3. 시각화 저장 경로 결정 로직
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
- `method` 와 `target_type` 을 정확히 구분해 폴더 구조를 자동 생성합니다.

### 4. CIFAR‑10 이미지 업샘플링
```python
def upsample_img(img_np, scale=4):
    """PIL bicubic 업샘플링으로 이미지를 scale 배 확대"""
    img_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    new_size = (pil_img.width * scale, pil_img.height * scale)
    pil_img = pil_img.resize(new_size, Image.BICUBIC)
    return np.array(pil_img).astype(np.float32) / 255.0
```
- 32×32 이미지를 4배(128×128) 확대해 시각적 품질을 크게 개선합니다.

### 5. 시각화 내용
- 원본 이미지, 공격 이미지, 그리고 **perturbation**(노이즈) 를 15배 확대해 3패널로 표시합니다.
- CIFAR‑10 클래스 번호를 실제 클래스 이름(airplane, cat 등)으로 매핑해 가독성을 높였습니다.

## 참고 사항
- **Targeted 공격 성공 기준**: 모델 예측이 지정한 목표 클래스와 일치해야 합니다.
- **Untargeted 공격 성공 기준**: 모델 예측이 원래 정답과 달라야 합니다.
- `results/` 폴더는 자동으로 생성되며, 기존 파일이 있으면 덮어쓰지 않고 새 폴더에 저장됩니다.

## 라이선스
학술 목적 사용을 전제로 합니다. 자유롭게 포크, 수정, 재배포 가능하지만 상업적 이용 시 별도 허가를 받아 주세요.
