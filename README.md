# trustworthy-ai-adversarial-attacks

# 신뢰할 수 있는 인공지능(1)

본 프로젝트는 신경망(Neural Networks) 모델의 취약점을 탐색하는 '적대적 공격(Adversarial Attacks)' 기법들을 MNIST와 CIFAR-10 데이터셋을 대상으로 검증하는 코드입니다. 구현된 공격 기법들은 다음과 같습니다:

* **Targeted FGSM** (Fast Gradient Sign Method)
* **Untargeted FGSM**
* **Targeted PGD** (Projected Gradient Descent)
* **Untargeted PGD**

## 파일 구성

* `test.py` : 전체 파이프라인 (데이터셋 다운로드, 모델 훈련 및 불러오기, 공격 수행 및 이미지 저장) 이 포함된 메인 스크립트입니다.
* `requirements.txt` : 프로그램 실행에 필요한 패키지들이 나열되어 있습니다. 
* `results/` (스크립트 실행 시 생성) : 공격 성공 지표 검증을 마친 각각의 $\epsilon$ (에러 허용치) 값에 대한 시각화 이미지가 저장되는 디렉토리입니다.
* `report.md` : 프로젝트 실험 분석 보고서 마크다운 파일입니다.

## 실행 환경 구성 및 실행 방법

### 1) 환경 설정 (Requirements Installation)

터미널에서 아래 명령어를 실행하여 필수 패키지를 설치해 주십시오.

```bash
pip install -r requirements.txt
```

### 2) 테스트 코드 실행 (Running the attacks)

아래 명령어를 통해 메인 스크립트를 실행합니다. 최초 실행 시 MNIST와 CIFAR-10 데이터셋, 그리고 Pre-trained ResNet-20 모델 가중치를 다운로드 받습니다.

```bash
python test.py
```

### 동작 과정

1. **MNIST 모델 학습**: 간단한 2계층 CNN을 정의하여 3 에폭(epochs) 간 학습하여 약 98% 이상의 정확도(Clean Accuracy)를 달성합니다.
2. **CIFAR-10 모델 로드**: `torch.hub`를 통해 `chenyaofo`의 사전 학습된 ResNet-20 모델 가중치를 다운로드받아 약 90% 이상의 깨끗한 환경에서의 정확도를 검증합니다.
3. **공격 지표 측정**: 공격 방식별로 $\epsilon \in \{0.05, 0.1, 0.2, 0.3\}$을 설정한 뒤, 각 샘플을 평가하여 공격 성공률 수치를 도출합니다.
4. **결과 시각화(저장)**: `results/` 디렉토리에 최대 $\epsilon$ (0.3) 에 대한 시각화 자료 (원본, 적대적 공격 이미지, 노이즈)를 각 기법과 데이터셋에 맞게 5장씩 생성 및 저장합니다.

## 주의 사항
* CPU 환경에서도 구동이 가능하지만(평가 시간 5-10분), GPU 환경(CUDA 지원 기기)에서 구동할 시 보다 빠른 동작 속도를 기대할 수 있습니다.
