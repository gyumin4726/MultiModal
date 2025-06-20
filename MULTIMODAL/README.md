# SCPC AI Challenge 2025 - Multimodal VQA Model

삼성 SCPC AI 챌린지를 위한 VMamba 기반 멀티모달 VQA(Visual Question Answering) 모델입니다.

## 프로젝트 구조

```
MULTIMODAL/
├── models/
│   ├── __init__.py
│   ├── vmamba_backbone.py       # VMamba 이미지 인코더
│   ├── text_encoder.py          # BERT 기반 텍스트 인코더
│   ├── fusion_module.py         # 멀티모달 퓨전 모듈
│   └── multimodal_classifier.py # 전체 멀티모달 분류기
├── datasets/
│   ├── __init__.py
│   └── vqa_dataset.py          # VQA 데이터셋 로더
├── configs/
│   ├── __init__.py
│   └── multimodal_config.py    # 모델 설정
├── utils/
│   ├── __init__.py
│   ├── data_utils.py           # 데이터 처리 유틸리티
│   └── training_utils.py       # 학습 유틸리티
├── train.py                    # 학습 스크립트
├── inference.py                # 추론 스크립트
└── requirements.txt            # 의존성 패키지
```

## 모델 아키텍처

1. **이미지 인코더**: VMamba 백본을 사용하여 이미지 특징 추출
2. **텍스트 인코더**: BERT를 사용하여 질문과 선택지 인코딩
3. **멀티모달 퓨전**: 크로스 어텐션을 통한 이미지-텍스트 특징 융합
4. **분류기**: 4개 선택지 중 정답 예측

## 사용법

### 학습
```bash
python train.py --config configs/multimodal_config.py
```

### 추론
```bash
python inference.py --model_path checkpoints/best_model.pth --test_csv dev_test.csv --output_csv submission.csv
```

## 대회 규칙 준수사항

- 단일 모델만 사용 (앙상블 불가)
- 모델 파라미터 수 < 3B
- 2024년 이전 공개된 사전학습 모델만 사용
- 외부 API 사용 금지 (모든 처리는 로컬에서 수행) 