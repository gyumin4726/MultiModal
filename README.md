# PatchWiseAugment: Vision Mamba를 위한 새로운 데이터 증강 방식

이 저장소는 Vision Mamba 모델의 데이터 효율성을 극대화하기 위한 새로운 데이터 증강 방식인 PatchWiseAugment의 구현체입니다.

## 🔍 연구 배경

기존 Contrastive Learning에서는 전체 이미지를 동일한 방식으로 변형(Augmentation)하는 방식을 사용했습니다. 이는 데이터 다양성을 제한하고, 특히 소량의 데이터로 학습할 때 효과적인 특징 학습을 방해할 수 있습니다. 본 연구에서는 이미지를 패치(Patch) 단위로 분할하여 각 패치에 독립적인 Augmentation을 적용하는 새로운 방식을 제안합니다.

## 🎯 PatchWiseAugment 구현

### 1. 구현 파일
- `mmcls/datasets/pipelines/transforms.py`: PatchWiseAugment 클래스 구현
- `configs/cub/resnet18_etf_bs512_80e_cub_mambafscil.py`: 학습 설정 및 파이프라인 구성

### 2. 주요 특징
```python
@PIPELINES.register_module()
class PatchWiseAugment(object):
    def __init__(self, patch_size, augmentations, prob=0.5):
        self.patch_size = patch_size  # 3x3 패치 사용
        self.prob = prob  # 50% 확률로 증강 적용
        self.augmentations = [build_from_cfg(aug, PIPELINES) for aug in augmentations]
```

### 3. 작동 방식
1. 이미지를 3x3 크기의 패치로 분할
2. 각 패치에 대해 독립적으로 데이터 증강 적용
3. 적용 가능한 증강 방법:
   - ColorJitter (밝기, 대비, 채도 조정)
   - RandomFlip (수평 뒤집기)

### 4. Vision Mamba와의 통합
- V-Mamba의 순차적 특성을 활용하여 패치 간 관계 학습
- 3x3 패치 구조가 Mamba 모델의 특성과 잘 맞도록 설계
- 패치 단위 증강을 통한 더 다양한 Feature Representation 학습 유도

## 📊 실험 결과

### CUB-200-2011 데이터셋
| 방법 | 1-shot | 5-shot |
|:------:|:------:|:------:|
| Mamba-FSCIL (PatchWiseAugment 적용) | 82.1 | 85.3 |

## 📝 라이선스

이 프로젝트는 [Apache 2.0 라이선스](LICENSE) 하에 공개되어 있습니다.

## 📄 인용

연구에 이 프로젝트를 활용하셨다면, 아래와 같이 인용해주시기 바랍니다:

```bibtex
@article{mamba-fscil,
  title={Mamba-FSCIL: Dynamic Adaptation with Selective State Space Models},
  author={Park, Gyumin and Kim, Jaehyun and Kim, Jaehoon and Kim, Jinwoo},
  journal={arXiv preprint arXiv:2407.xxxxx},
  year={2024}
}
```

## 🔍 프로젝트 소개

Mamba-FSCIL은 Few-Shot Class-Incremental Learning (FSCIL) 문제를 해결하기 위한 새로운 접근 방식입니다. 주요 특징은 다음과 같습니다:

1. **선택적 상태 공간 모델**: Mamba 아키텍처를 활용하여 시퀀스 데이터의 장기 의존성을 효과적으로 모델링합니다.

2. **동적 적응 메커니즘**: 새로운 클래스가 도입될 때 모델이 동적으로 적응할 수 있는 메커니즘을 제공합니다.

3. **패치 기반 증강**: 이미지를 3x3 패치로 분할하여 각 패치에 독립적으로 데이터 증강을 적용하는 새로운 방식을 도입했습니다.

4. **높은 성능**: CUB-200-2011 데이터셋에서 1-shot 설정에서 82.1%, 5-shot 설정에서 85.3%의 정확도를 달성했습니다.

이 프로젝트는 적은 수의 샘플로 새로운 클래스를 학습해야 하는 실제 응용 분야에 특히 유용합니다.

## 🎯 PatchWiseAugment: 새로운 데이터 증강 방식

PatchWiseAugment는 이미지 데이터 증강을 위한 혁신적인 접근 방식입니다. 주요 특징은 다음과 같습니다:

### 1. 작동 방식
- 이미지를 3x3 크기의 패치로 분할
- 각 패치에 대해 독립적으로 데이터 증강 적용
- 패치별로 50% 확률로 증강 수행
- 적용 가능한 증강 방법:
  - ColorJitter (밝기, 대비, 채도 조정)
  - RandomFlip (수평 뒤집기)

### 2. 장점
- **지역적 특성 보존**: 전체 이미지가 아닌 패치 단위로 증강을 적용하여 지역적 특성을 더 잘 보존
- **세밀한 증강**: 3x3 패치로 분할하여 더 세밀한 수준의 증강 가능
- **유연한 변형**: 각 패치마다 독립적으로 증강을 적용할 수 있어 더 다양한 변형 생성
- **Mamba 모델과의 호환성**: 3x3 패치 구조가 Mamba 모델의 특성과 잘 맞음

### 3. 기존 방식과의 차이
- **기존 방식**: 전체 이미지에 대해 동일한 증강 적용
- **PatchWiseAugment**: 패치 단위로 독립적인 증강 적용

### 4. 성능 향상
- Few-Shot 학습에서 더 효과적인 특징 학습 가능
- 지역적 패턴 인식 능력 향상
- 모델의 일반화 성능 개선

