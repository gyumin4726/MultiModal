# DirectionalPatchAugment 설명

## 개요
DirectionalPatchAugment는 이미지 패치 단위로 방향성 있는 증강을 수행하는 데이터 증강 기법입니다.

## 주요 특징
- 4가지 방향별로 서로 다른 증강 효과 적용
  - SE(↘): 채도(saturation) 조정
  - NE(↗): 대비(contrast) 조정
  - SW(↙): 밝기(brightness) 조정
  - NW(↖): 블러(blur) 조정

## 구현 세부사항
1. **증강 강도 제어**
   - `strength` 파라미터로 각 효과의 강도 조절 (기본값: 0.5)
   - 값이 클수록 더 강한 증강 효과 적용

2. **시각화 기능**
   - `visualize=True`로 설정 시 증강 결과 저장
   - 각 이미지는 고유 ID로 저장됨 (클래스ID * 10000 + 이미지ID)
   - 저장 경로: `work_dirs/directional_vis/aug_{img_id:06d}.jpg`

3. **파이프라인 적용**
   ```python
   train_pipeline = [
       dict(type='LoadImageFromFile'),
       dict(type='Resize', size=(_img_resize_size, _img_resize_size)),
       dict(type='CenterCrop', crop_size=img_size),
       dict(type='DirectionalPatchAugment',
            patch_size=7,
            strength=0.5,
            visualize=True,
            vis_dir='work_dirs/directional_vis'),
       # ... 기타 파이프라인 단계들
   ]
   ```

## 주의사항
1. **학습/평가 구분**
   - 데이터 증강은 학습(training) 과정에서만 사용
   - 평가(evaluation) 과정에서는 적용하지 않음
   - 테스트 시에는 원본 이미지 그대로 사용하여 공정한 평가 수행

2. **시각화 저장**
   - 클래스ID와 이미지ID를 조합하여 고유한 파일명 생성
   - 이를 통해 증강된 이미지의 중복 저장 방지
   - 시각화 결과물로 증강 효과 확인 가능

## 사용 예시
```python
# 설정 예시
dict(type='DirectionalPatchAugment',
     patch_size=7,          # 패치 크기
     strength=0.5,          # 증강 강도
     visualize=True,        # 시각화 활성화
     vis_dir='work_dirs/directional_vis')  # 저장 경로
``` 