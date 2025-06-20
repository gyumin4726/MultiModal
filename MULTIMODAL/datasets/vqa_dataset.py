import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional, Union
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class VQADataset(Dataset):
    """VQA 데이터셋 클래스
    
    SCPC AI Challenge의 VQA 데이터를 로드하고 전처리합니다.
    
    Args:
        csv_path (str): CSV 파일 경로
        image_dir (str): 이미지 디렉토리 경로
        transform (callable, optional): 이미지 변환 함수
        mode (str): 데이터셋 모드 ('train', 'test')
    """
    
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform: Optional[callable] = None,
        mode: str = 'train'
    ):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.mode = mode
        
        # CSV 데이터 로드
        self.data = pd.read_csv(csv_path)
        print(f"Loaded {len(self.data)} samples from {csv_path}")
        
        # 이미지 변환 설정
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
        
        # 데이터 검증
        self._validate_data()
    
    def _get_default_transform(self) -> transforms.Compose:
        """기본 이미지 변환 함수"""
        if self.mode == 'train':
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _validate_data(self):
        """데이터 유효성 검사"""
        required_columns = ['ID', 'img_path', 'Question', 'A', 'B', 'C', 'D']
        
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 이미지 파일 존재 확인 (샘플링)
        sample_size = min(10, len(self.data))
        for i in range(sample_size):
            img_path = os.path.join(self.image_dir, self.data.iloc[i]['img_path'])
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
        
        print(f"Data validation completed for {self.mode} mode")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """데이터 샘플 반환
        
        Args:
            idx (int): 샘플 인덱스
            
        Returns:
            Dict: 샘플 데이터
                - 'image': 이미지 텐서 (3, 224, 224)
                - 'question': 질문 문자열
                - 'choices': 선택지 리스트 [A, B, C, D]
                - 'answer': 정답 (train 모드에서만, 0-3)
                - 'id': 샘플 ID
        """
        row = self.data.iloc[idx]
        
        # 이미지 로드
        img_path = os.path.join(self.image_dir, row['img_path'])
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 더미 이미지 생성
            image = torch.zeros(3, 224, 224)
        
        # 텍스트 데이터
        question = str(row['Question'])
        choices = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
        
        sample = {
            'image': image,
            'question': question,
            'choices': choices,
            'choice_a': str(row['A']),
            'choice_b': str(row['B']),
            'choice_c': str(row['C']),
            'choice_d': str(row['D']),
            'id': str(row['ID'])
        }
        
        # 정답 정보 (train 모드에서만)
        if 'answer' in row and pd.notna(row['answer']):
            answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            sample['answer'] = answer_map.get(str(row['answer']).upper(), 0)
        
        return sample
    
    def get_sample_by_id(self, sample_id: str) -> Dict:
        """ID로 샘플 검색"""
        mask = self.data['ID'] == sample_id
        if not mask.any():
            raise ValueError(f"Sample ID {sample_id} not found")
        
        idx = mask.idxmax()
        return self.__getitem__(idx)
    
    def get_statistics(self) -> Dict[str, Union[int, float, List]]:
        """데이터셋 통계 정보"""
        stats = {
            'total_samples': len(self.data),
            'unique_questions': self.data['Question'].nunique(),
            'avg_question_length': self.data['Question'].str.len().mean(),
            'avg_choice_length': {
                'A': self.data['A'].str.len().mean(),
                'B': self.data['B'].str.len().mean(),
                'C': self.data['C'].str.len().mean(),
                'D': self.data['D'].str.len().mean()
            }
        }
        
        if 'answer' in self.data.columns:
            answer_dist = self.data['answer'].value_counts().to_dict()
            stats['answer_distribution'] = answer_dist
        
        return stats


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """배치 데이터 콜레이션 함수
    
    Args:
        batch (List[Dict]): 배치 데이터
        
    Returns:
        Dict: 콜레이션된 배치 데이터
    """
    # 이미지 스택
    images = torch.stack([sample['image'] for sample in batch])
    
    # 텍스트 데이터 리스트로 수집
    questions = [sample['question'] for sample in batch]
    choices_a = [sample['choice_a'] for sample in batch]
    choices_b = [sample['choice_b'] for sample in batch]
    choices_c = [sample['choice_c'] for sample in batch]
    choices_d = [sample['choice_d'] for sample in batch]
    ids = [sample['id'] for sample in batch]
    
    batch_data = {
        'images': images,
        'questions': questions,
        'choices_a': choices_a,
        'choices_b': choices_b,
        'choices_c': choices_c,
        'choices_d': choices_d,
        'ids': ids
    }
    
    # 정답이 있는 경우 추가
    if 'answer' in batch[0]:
        answers = torch.tensor([sample['answer'] for sample in batch], dtype=torch.long)
        batch_data['answers'] = answers
    
    return batch_data


class VQADataLoader:
    """VQA 데이터로더 래퍼 클래스"""
    
    @staticmethod
    def create_train_loader(
        csv_path: str,
        image_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        transform: Optional[callable] = None
    ) -> DataLoader:
        """학습용 데이터로더 생성"""
        dataset = VQADataset(
            csv_path=csv_path,
            image_dir=image_dir,
            transform=transform,
            mode='train'
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True
        )
    
    @staticmethod
    def create_test_loader(
        csv_path: str,
        image_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        transform: Optional[callable] = None
    ) -> DataLoader:
        """테스트용 데이터로더 생성"""
        dataset = VQADataset(
            csv_path=csv_path,
            image_dir=image_dir,
            transform=transform,
            mode='test'
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )


def test_vqa_dataset():
    """VQA 데이터셋 테스트"""
    print("Testing VQA Dataset...")
    
    # 더미 CSV 데이터 생성
    dummy_data = {
        'ID': ['TEST_000', 'TEST_001'],
        'img_path': ['TEST_000.jpg', 'TEST_001.jpg'],
        'Question': ['What color is the bird?', 'How many people are there?'],
        'A': ['Red', 'One'],
        'B': ['Blue', 'Two'],
        'C': ['Green', 'Three'],
        'D': ['Yellow', 'Four'],
        'answer': ['A', 'B']
    }
    
    # 임시 CSV 파일 생성
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df = pd.DataFrame(dummy_data)
        df.to_csv(f.name, index=False)
        csv_path = f.name
    
    # 임시 이미지 디렉토리
    image_dir = tempfile.mkdtemp()
    
    try:
        # 더미 이미지 생성
        for img_name in ['TEST_000.jpg', 'TEST_001.jpg']:
            img_path = os.path.join(image_dir, img_name)
            dummy_img = Image.new('RGB', (224, 224), color='red')
            dummy_img.save(img_path)
        
        # 데이터셋 테스트
        dataset = VQADataset(csv_path, image_dir, mode='train')
        
        print(f"Dataset length: {len(dataset)}")
        print(f"Dataset statistics: {dataset.get_statistics()}")
        
        # 샘플 테스트
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Question: {sample['question']}")
        print(f"Choices: {sample['choices']}")
        
        # 데이터로더 테스트
        dataloader = VQADataLoader.create_train_loader(
            csv_path, image_dir, batch_size=2, num_workers=0
        )
        
        for batch in dataloader:
            print(f"Batch keys: {batch.keys()}")
            print(f"Images shape: {batch['images'].shape}")
            print(f"Questions: {batch['questions']}")
            print(f"Answers shape: {batch['answers'].shape}")
            break
        
        print("VQA Dataset test passed!")
        
    finally:
        # 임시 파일 정리
        os.unlink(csv_path)
        import shutil
        shutil.rmtree(image_dir)


if __name__ == "__main__":
    test_vqa_dataset() 