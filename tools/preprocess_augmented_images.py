import os
import sys
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from mmcls.datasets import build_dataset
from mmcls.datasets.pipelines.directional_patch_augment import DirectionalPatchAugment
from mmfscil.datasets.cub import CUBFSCILDataset

def preprocess_and_save_augmented_images(config_file, output_dir):
    """증강된 이미지를 미리 생성하고 저장합니다."""
    # 설정 파일 로드
    cfg = mmcv.Config.fromfile(config_file)
    
    # 전처리용 파이프라인 정의
    preprocess_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='DirectionalPatchAugment',
             patch_size=7,
             strength=0.5,
             visualize=False)
    ]
    
    # 데이터셋 설정 수정
    cfg.data.train.dataset.pipeline = preprocess_pipeline
    
    # 데이터셋 생성
    dataset = build_dataset(cfg.data.train.dataset)
    
    # 원본 이미지 수 계산
    original_counts = {}
    for idx in range(len(dataset)):
        results = dataset[idx]
        class_name = os.path.basename(os.path.dirname(results['img_info']['filename']))
        original_counts[class_name] = original_counts.get(class_name, 0) + 1
    
    print('\nOriginal images per class:')
    for class_name, count in sorted(original_counts.items()):
        print(f'{class_name}: {count}')
    
    # 출력 디렉토리 생성
    mmcv.mkdir_or_exist(output_dir)
    
    # 각 이미지에 대해 증강 적용
    augmented_counts = {}
    for idx in range(len(dataset)):
        # 이미지 로드 및 증강
        results = dataset[idx]
        
        # 증강된 이미지 저장
        img = results['img']
        if isinstance(img, DC):
            img = img.data
            
        # 원본 이미지 경로에서 클래스 정보 추출
        original_path = results['img_info']['filename']
        class_name = os.path.basename(os.path.dirname(original_path))
        image_name = os.path.basename(original_path)
        
        # 클래스별 디렉토리 생성
        class_dir = os.path.join(output_dir, class_name)
        mmcv.mkdir_or_exist(class_dir)
        
        # 저장 경로 설정 (원본과 동일한 이름으로 저장)
        save_path = os.path.join(class_dir, image_name)
        
        # 이미지 저장
        mmcv.imwrite(img, save_path)
        augmented_counts[class_name] = augmented_counts.get(class_name, 0) + 1
        
        if idx % 100 == 0:
            print(f'Processed {idx}/{len(dataset)} images')
    
    # 증강된 이미지 수 출력 및 비교
    print('\nAugmented images per class:')
    print('Class name: Original -> Augmented (Difference)')
    for class_name in sorted(original_counts.keys()):
        orig_count = original_counts[class_name]
        aug_count = augmented_counts.get(class_name, 0)
        diff = orig_count - aug_count
        print(f'{class_name}: {orig_count} -> {aug_count} ({diff:+d})')

if __name__ == '__main__':
    config_file = 'configs/cub/resnet18_etf_bs512_80e_cub_mambafscil.py'
    output_dir = 'data/CUB_200_2011/augmented_images'
    preprocess_and_save_augmented_images(config_file, output_dir) 