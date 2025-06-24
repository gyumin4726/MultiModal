"""
MultiModal VQA Model Package

이 패키지는 VQA(Visual Question Answering)를 위한 
멀티모달 AI 모델들을 포함합니다.

Components:
- vision_encoder: 이미지 특징 추출
- text_encoder: 텍스트 특징 추출  
- language_model: 언어 모델
- multimodal_fusion: 멀티모달 융합
- model: 통합 모델
"""

from . import vision_encoder
from . import text_encoder
from . import language_model
from . import multimodal_fusion
from . import model

__version__ = "1.0.0"
__author__ = "MultiModal VQA Team" 