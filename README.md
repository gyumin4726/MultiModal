# 🌐 MultiModal AI System

**차세대 멀티모달 이미지-텍스트 질의응답 시스템**

이 프로젝트는 기존 BLIP2 기반 시스템을 완전히 대체하여, VMamba 비전 인코더, sentence-transformers 텍스트 인코더, 그리고 microsoft/phi-2 언어 모델을 활용한 새로운 멀티모달 AI 시스템입니다.

---

## 📋 목차

- [시스템 개요](#-시스템-개요)
- [아키텍처](#-아키텍처)
- [프로젝트 구조](#-프로젝트-구조)
- [모듈 설명](#-모듈-설명)
- [설치 및 실행](#-설치-및-실행)
- [현재 상태](#-현재-상태)
- [사용 방법](#-사용-방법)
- [개발 로드맵](#-개발-로드맵)
- [문제점 및 해결 방안](#-문제점-및-해결-방안)

---

## 🎯 시스템 개요

### **핵심 특징**
- ✅ **완전 모듈화**: 각 구성요소가 독립적으로 관리됨
- ✅ **사전학습 모델 활용**: 모든 컴포넌트가 사전학습된 모델 사용
- ✅ **최신 아키텍처**: VMamba(비전) + BERT계열(텍스트) + phi-2(LLM)
- ⚠️ **현재 상태**: 기본 추론 가능, 완전한 멀티모달 융합은 구현 중

### **대상 태스크**
- 이미지 기반 다중 선택 질의응답
- 시각적 추론 및 이해
- 멀티모달 컨텍스트 분석

---

## 🏗️ 아키텍처

```
📸 이미지 입력 (224×224×3)
    ↓
🔍 VMamba 비전 인코더
    ↓
📊 768차원 비전 피처

📝 텍스트 질문
    ↓  
🔤 sentence-transformers 텍스트 인코더
    ↓
📊 768차원 텍스트 피처

     ↓ (융합 - 구현 중)
🔗 Cross-attention 멀티모달 융합
     ↓
🧠 microsoft/phi-2 언어 모델
     ↓
📋 최종 답변 (A/B/C/D)
```

---

## 📁 프로젝트 구조

```
MultiModal/
├── 📄 README.md                # 현재 파일
├── 📄 config.py                # 환경 설정 및 시드 고정
├── 📄 utils.py                 # 유틸리티 함수들
│
├── 🔧 **핵심 모듈들**
├── 📄 vision_encoder.py        # VMamba 기반 비전 인코더
├── 📄 text_encoder.py          # sentence-transformers 기반 텍스트 인코더  
├── 📄 language_model.py        # microsoft/phi-2 언어 모델
├── 📄 model.py                 # 통합 import 모듈
│
├── 🚀 **실행 파일들**
├── 📄 main.py                  # 개별 모듈 테스트
├── 📄 inference.py             # 전체 추론 시스템
├── 📄 multimodal_fusion.py     # 멀티모달 융합 (구현 중)
│
├── 📊 **데이터**
├── 📄 dev_test.csv             # 테스트 데이터 (60개 샘플)
├── 📄 sample_submission.csv    # 제출 형식
├── 📁 input_images/            # 테스트 이미지들 (TEST_000.jpg ~ TEST_059.jpg)
│
└── 📈 **결과**
    └── 📄 multimodal_submission.csv  # 추론 결과 (생성됨)
```

---

## 🔧 모듈 설명

### **1️⃣ 비전 인코더 (`vision_encoder.py`)**
- **모델**: VMamba (`vmamba_tiny_s1l8`)
- **입력**: RGB 이미지 (224×224×3)
- **출력**: 768차원 비전 피처 벡터
- **특징**: 
  - 사전학습된 VMamba 백본 사용
  - MambaNeck을 통한 피처 변환
  - 멀티스케일 스킵 연결 지원

```python
from vision_encoder import load_vision_encoder

vision_encoder = load_vision_encoder(
    model_name='vmamba_tiny_s1l8',
    output_dim=768
)
features = vision_encoder(image_tensor)  # (1, 768)
```

### **2️⃣ 텍스트 인코더 (`text_encoder.py`)**
- **모델**: sentence-transformers (`all-MiniLM-L6-v2`)
- **입력**: 텍스트 문자열
- **출력**: 768차원 텍스트 피처 벡터
- **특징**:
  - 사전학습된 BERT 계열 모델
  - 다양한 모델 옵션 지원 (다국어, 고성능)
  - L2 정규화 및 projection layer

```python
from text_encoder import load_text_encoder

text_encoder = load_text_encoder(model_type='default')
features = text_encoder(["What is in this image?"])  # (1, 768)
```

### **3️⃣ 언어 모델 (`language_model.py`)**
- **모델**: microsoft/phi-2 (2.7B 파라미터)
- **기능**: 텍스트 생성 및 추론
- **특징**:
  - 사전학습된 고성능 소형 LLM
  - 효율적인 추론 성능
  - 멀티태스크 지원

```python
from language_model import load_language_model

llm = load_language_model()
response = llm.generate_text("Question: ...", max_new_tokens=10)
```

---

## 🚀 설치 및 실행

### **의존성 설치**
```bash
# 필수 패키지들
pip install torch torchvision transformers
pip install sentence-transformers  # 텍스트 인코더용
pip install pandas pillow tqdm

# VMamba 관련 (필요시)
pip install mmcv-full  # 또는 mmcv
```

### **개별 모듈 테스트**
```bash
cd MultiModal
python main.py
```

### **전체 추론 실행**
```bash
cd MultiModal  
python inference.py
```

---

## 📊 현재 상태

### **✅ 완료된 것들**
- [x] 모든 모듈 구현 및 분리
- [x] VMamba 비전 인코더 구현
- [x] sentence-transformers 텍스트 인코더 구현
- [x] microsoft/phi-2 언어 모델 통합
- [x] 기본 추론 파이프라인 구현
- [x] 데이터 로딩 및 전처리
- [x] 결과 저장 시스템

### **⚠️ 진행 중인 것들**
- [ ] 완전한 멀티모달 융합 적용
- [ ] Cross-attention 메커니즘 통합
- [ ] 의존성 문제 해결
- [ ] 성능 최적화

### **❌ 현재 문제점**
1. **의존성 미설치**: `sentence-transformers`, `mmcv` 설치 필요
2. **융합 미적용**: 비전/텍스트 피처를 추출하지만 실제 융합은 안 됨
3. **VMamba 경로**: 사전학습 가중치 경로 확인 필요

---

## 💻 사용 방법

### **1. 개별 모듈 테스트**
```python
# main.py 실행으로 각 모듈 상태 확인
python main.py

# 출력 예시:
# 🔍 Loading vision encoder...
# ✅ Vision encoder loaded successfully!
# 📝 Loading text encoder...  
# ✅ Text encoder loaded successfully!
# 🤖 Loading language model...
# ✅ Language model loaded successfully!
```

### **2. 전체 시스템 추론**
```python
# inference.py로 dev_test.csv 전체 추론
python inference.py

# 결과: multimodal_submission.csv 생성
```

### **3. 개별 함수 사용**
```python
from vision_encoder import load_vision_encoder
from text_encoder import load_text_encoder
from language_model import load_language_model

# 모델 로딩
vision_encoder = load_vision_encoder()
text_encoder = load_text_encoder()
llm = load_language_model()

# 추론
vision_features = vision_encoder(image_tensor)
text_features = text_encoder(["What is this?"])
response = llm.generate_text("Based on the image...")
```

---

## 🛤️ 개발 로드맵

### **Phase 1: 기본 시스템 구축** ✅
- [x] 모듈 분리 및 구현
- [x] 각 인코더 구현
- [x] LLM 통합

### **Phase 2: 멀티모달 융합** 🚧 (진행 중)
- [ ] Cross-attention 메커니즘 적용
- [ ] 융합된 피처를 활용한 추론
- [ ] 성능 평가 및 최적화

### **Phase 3: 고도화** 📋 (예정)
- [ ] 더 복잡한 융합 아키텍처
- [ ] 파인튜닝 지원
- [ ] 배치 처리 최적화
- [ ] 추론 속도 개선

---

## ❗ 문제점 및 해결 방안

### **1. 의존성 문제**
```bash
# 에러: ModuleNotFoundError: No module named 'sentence_transformers'
# 해결: pip install sentence-transformers

# 에러: No module named 'mmcv'  
# 해결: pip install mmcv-full
```

### **2. 멀티모달 융합 미적용**
**현재 상황**: 
- 비전 피처 추출: ✅
- 텍스트 피처 추출: ✅  
- 융합 활용: ❌ (텍스트 프롬프트만 사용)

**해결 방안**:
```python
# multimodal_fusion.py에서 제안한 방식 적용
from multimodal_fusion import EnhancedMultiModalProcessor

processor = EnhancedMultiModalProcessor(vision_encoder, text_encoder, llm)
response = processor.process_multimodal_input(image, question, choices)
```

### **3. VMamba 가중치 경로**
```python
# 현재: './vssm1_tiny_0230s_ckpt_epoch_264.pth'
# 확인 필요: 실제 파일이 존재하는지 체크
```

---

## 📈 성능 및 특징

### **모델 크기 및 성능**
| 모듈 | 모델 | 파라미터 | 메모리 사용량 |
|------|------|----------|---------------|
| 비전 | VMamba-tiny | ~22M | ~1GB |
| 텍스트 | MiniLM-L6 | ~22M | ~500MB |
| LLM | phi-2 | 2.7B | ~6GB |
| **총합** | - | **~2.7B** | **~7.5GB** |

### **장점**
- 🚀 **효율성**: 상대적으로 작은 모델 크기
- 🔧 **모듈화**: 각 구성요소 독립 관리
- 📚 **사전학습**: 별도 학습 없이 바로 사용
- 🎯 **전문성**: 각 도메인에 최적화된 모델 사용

---

## 🤝 기여 방법

1. **Issue 리포팅**: 버그나 개선사항 제안
2. **코드 기여**: 새로운 기능이나 최적화
3. **문서화**: README나 코드 주석 개선
4. **테스트**: 다양한 환경에서의 테스트

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 제공됩니다.

---

## 📞 연락처

프로젝트 관련 문의나 협업 제안은 이슈를 통해 연락해주세요.

---

*마지막 업데이트: 2024년 12월* 