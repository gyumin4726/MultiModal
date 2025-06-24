import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class VQATextEncoder(nn.Module):
    """VQA 태스크 특화 텍스트 인코더 - Question과 Choices를 구조적으로 처리"""
    
    def __init__(self, 
                 model_name='sentence-transformers/all-mpnet-base-v2',
                 output_dim=1024,
                 device=None,
                 max_length=256):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 고성능 transformer 모델 로딩
        print(f"Loading VQA-optimized text encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.native_dim = self.model.config.hidden_size
        print(f"Native embedding dimension: {self.native_dim}")
        
        # VQA 특화 구조
        # 1. Question 인코더
        self.question_proj = nn.Linear(self.native_dim, output_dim // 2).to(self.device)
        
        # 2. Choices 인코더 (4개 선택지)
        self.choice_proj = nn.Linear(self.native_dim, output_dim // 8).to(self.device)  # 각 선택지당 128차원
        
        # 3. Question-Choice 상호작용 모듈
        self.qc_attention = nn.MultiheadAttention(
            embed_dim=output_dim // 8,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        ).to(self.device)
        
        # 4. 최종 융합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim // 2 + output_dim // 2, output_dim),  # question + attended_choices
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        ).to(self.device)
        
        print(f"VQA Text Encoder initialized: {self.native_dim} -> {output_dim}")
    
    def encode_single_text(self, text):
        """단일 텍스트 인코딩"""
        if isinstance(text, str):
            text = [text]
        
        encoded_input = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            # CLS token 또는 mean pooling
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def encode_vqa_structured(self, question, choices):
        """VQA 구조화 인코딩 - Question과 Choices를 따로 처리 후 융합"""
        
        # 1. Question 인코딩
        question_emb = self.encode_single_text(question)  # (1, native_dim)
        question_feat = self.question_proj(question_emb)  # (1, output_dim//2)
        
        # 2. 각 Choice 인코딩
        choice_embeddings = []
        for choice in choices:
            choice_emb = self.encode_single_text(choice)  # (1, native_dim)
            choice_feat = self.choice_proj(choice_emb)    # (1, output_dim//8)
            choice_embeddings.append(choice_feat)
        
        # 3. Choices를 하나의 텐서로 결합
        choices_tensor = torch.stack(choice_embeddings, dim=1)  # (1, 4, output_dim//8)
        
        # 4. Question을 query로, Choices를 key/value로 attention
        question_query = question_feat[:, :self.output_dim//8].unsqueeze(1)  # (1, 1, output_dim//8)
        
        attended_choices, attention_weights = self.qc_attention(
            question_query, choices_tensor, choices_tensor
        )  # (1, 1, output_dim//8)
        
        # 5. 모든 choice 정보를 결합
        all_choices = choices_tensor.flatten(start_dim=1)  # (1, 4 * output_dim//8)
        
        # 6. Question + Attended Choices 융합
        final_input = torch.cat([
            question_feat,  # (1, output_dim//2)
            all_choices     # (1, output_dim//2)
        ], dim=-1)
        
        final_features = self.fusion_layer(final_input)  # (1, output_dim)
        
        return final_features, attention_weights.squeeze()
    
    def forward(self, question, choices):
        """VQA 텍스트 인코딩"""
        return self.encode_vqa_structured(question, choices)


class ContextualTextEncoder(nn.Module):
    """컨텍스트 강화 텍스트 인코더 - 질문 유형별 특화"""
    
    def __init__(self, base_encoder):
        super().__init__()
        self.base_encoder = base_encoder
        self.output_dim = base_encoder.output_dim
        
        # 질문 유형별 컨텍스트 임베딩
        self.context_embeddings = nn.Embedding(10, self.output_dim // 4).to(base_encoder.device)  # 10가지 질문 유형
        
        # 컨텍스트 융합 레이어
        self.context_fusion = nn.Sequential(
            nn.Linear(self.output_dim + self.output_dim // 4, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        ).to(base_encoder.device)
        
        # 질문 유형 분류기
        self.question_type_map = {
            'visual_attributes': 0,
            'object_scene_recognition': 1,
            'activity_behavior': 2,
            'inference_reasoning': 3,
            'contextual_understanding': 4,
            'knowledge_based': 5,
            'comparison_selection': 6,
            'quantitative_analysis': 7,
            'general_visual_qa': 8,
            'default': 9
        }
    
    def get_question_type_id(self, question):
        """질문 유형 ID 반환"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['color', 'size', 'shape']):
            return self.question_type_map['visual_attributes']
        elif any(word in question_lower for word in ['what is', 'what are', 'identify']):
            return self.question_type_map['object_scene_recognition']
        elif any(word in question_lower for word in ['doing', 'activity', 'behavior']):
            return self.question_type_map['activity_behavior']
        elif any(word in question_lower for word in ['might', 'likely', 'purpose']):
            return self.question_type_map['inference_reasoning']
        elif any(word in question_lower for word in ['where', 'when', 'time']):
            return self.question_type_map['contextual_understanding']
        elif any(word in question_lower for word in ['common', 'typical', 'culture']):
            return self.question_type_map['knowledge_based']
        elif any(word in question_lower for word in ['which', 'best', 'most']):
            return self.question_type_map['comparison_selection']
        elif any(word in question_lower for word in ['how many', 'count']):
            return self.question_type_map['quantitative_analysis']
        else:
            return self.question_type_map['general_visual_qa']
    
    def forward(self, question, choices):
        """컨텍스트 강화 인코딩"""
        # 1. 기본 VQA 인코딩
        base_features, attention_weights = self.base_encoder(question, choices)
        
        # 2. 질문 유형별 컨텍스트 추가
        question_type_id = self.get_question_type_id(question)
        context_emb = self.context_embeddings(torch.tensor(question_type_id).to(base_features.device))
        context_emb = context_emb.unsqueeze(0)  # (1, output_dim//4)
        
        # 3. 컨텍스트 융합
        enhanced_input = torch.cat([base_features, context_emb], dim=-1)
        enhanced_features = self.context_fusion(enhanced_input)
        
        return enhanced_features, attention_weights


def load_vqa_text_encoder(model_type='vqa_optimized', **kwargs):
    """VQA 특화 텍스트 인코더 로딩"""
    
    if model_type == 'vqa_optimized':
        base_encoder = VQATextEncoder(**kwargs)
        return ContextualTextEncoder(base_encoder)
    else:
        return VQATextEncoder(**kwargs)


if __name__ == "__main__":
    print("🚀 VQA-Optimized Text Encoder Test")
    print("="*50)
    
    # VQA 특화 인코더 테스트
    encoder = load_vqa_text_encoder('vqa_optimized', output_dim=1024)
    
    test_question = "What color is the car in the image?"
    test_choices = ["Red", "Blue", "Green", "Yellow"]
    
    features, attention = encoder(test_question, test_choices)
    print(f"✅ VQA Features shape: {features.shape}")
    print(f"✅ Attention weights: {attention.shape}")
    print(f"✅ Question-Choice attention: {attention}") 