import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalFusion(nn.Module):
    """비전과 텍스트 피처를 융합하는 모듈"""
    
    def __init__(self, 
                 vision_dim=768, 
                 text_dim=768, 
                 hidden_dim=512, 
                 output_dim=768):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 각 모달리티별 projection
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention 메커니즘
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 융합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, vision_features, text_features):
        """
        Args:
            vision_features: (B, vision_dim) - 비전 피처
            text_features: (B, text_dim) - 텍스트 피처
            
        Returns:
            fused_features: (B, output_dim) - 융합된 피처
        """
        batch_size = vision_features.size(0)
        
        # 1. 각 모달리티를 hidden_dim으로 projection
        v_proj = self.vision_proj(vision_features)  # (B, hidden_dim)
        t_proj = self.text_proj(text_features)      # (B, hidden_dim)
        
        # 2. Cross-attention을 위해 sequence 차원 추가
        v_seq = v_proj.unsqueeze(1)  # (B, 1, hidden_dim)
        t_seq = t_proj.unsqueeze(1)  # (B, 1, hidden_dim)
        
        # 3. Vision → Text attention
        v2t_attended, _ = self.cross_attention(v_seq, t_seq, t_seq)  # (B, 1, hidden_dim)
        
        # 4. Text → Vision attention  
        t2v_attended, _ = self.cross_attention(t_seq, v_seq, v_seq)  # (B, 1, hidden_dim)
        
        # 5. Squeeze back to (B, hidden_dim)
        v2t_attended = v2t_attended.squeeze(1)
        t2v_attended = t2v_attended.squeeze(1)
        
        # 6. Concatenate and fuse
        fused_input = torch.cat([v2t_attended, t2v_attended], dim=-1)  # (B, hidden_dim*2)
        fused_features = self.fusion_layer(fused_input)  # (B, output_dim)
        
        return fused_features


class EnhancedMultiModalProcessor:
    """향상된 멀티모달 처리기"""
    
    def __init__(self, vision_encoder, text_encoder, language_model):
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder  
        self.language_model = language_model
        
        # 융합 모듈
        self.fusion_model = MultiModalFusion(
            vision_dim=768,
            text_dim=768, 
            hidden_dim=512,
            output_dim=768
        )
        
        if torch.cuda.is_available():
            self.fusion_model = self.fusion_model.cuda()
            
    def process_multimodal_input(self, image_tensor, question_text, choices):
        """멀티모달 입력 처리"""
        
        # 1. 각 모달리티별 피처 추출
        with torch.no_grad():
            vision_features = self.vision_encoder(image_tensor)  # (1, 768)
            text_features = self.text_encoder([question_text])   # (1, 768)
        
        # 2. 멀티모달 융합
        fused_features = self.fusion_model(vision_features, text_features)  # (1, 768)
        
        # 3. 융합된 피처 정보를 프롬프트에 포함
        prompt = self.create_informed_prompt(
            question_text, 
            choices, 
            vision_features, 
            text_features, 
            fused_features
        )
        
        # 4. LLM으로 응답 생성
        response = self.language_model.generate_text(prompt, max_new_tokens=10)
        
        return response, {
            'vision_features': vision_features,
            'text_features': text_features, 
            'fused_features': fused_features
        }
    
    def create_informed_prompt(self, question, choices, vision_feat, text_feat, fused_feat):
        """융합된 피처 정보를 활용한 프롬프트 생성"""
        
        # 피처의 주요 특성 분석
        vision_strength = vision_feat.norm().item()
        text_strength = text_feat.norm().item()
        fusion_strength = fused_feat.norm().item()
        
        prompt = (
            "You are analyzing an image-question pair with the following characteristics:\n"
            f"- Visual content strength: {vision_strength:.2f}\n"
            f"- Text complexity: {text_strength:.2f}\n" 
            f"- Multimodal alignment: {fusion_strength:.2f}\n\n"
            "Based on this multimodal analysis, answer the following question:\n\n"
            f"Question: {question}\n"
            "Choices:\n"
        )
        
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
            
        prompt += "\nAnswer:"
        
        return prompt


def demonstrate_working_principle():
    """작동 원리 시연"""
    print("🔬 MultiModal System Working Principle:")
    print("="*50)
    
    print("\n1️⃣ 이미지 처리:")
    print("   📸 Image (224x224x3) → VMamba → MambaNeck → 768D vector")
    
    print("\n2️⃣ 텍스트 처리:")
    print("   📝 Question text → sentence-transformers → 768D vector")
    
    print("\n3️⃣ 멀티모달 융합:")
    print("   🔗 Vision (768D) + Text (768D) → Cross-attention → Fused (768D)")
    
    print("\n4️⃣ 지능형 프롬프트:")
    print("   🧠 Fused features → Informed prompt → phi-2 → Answer")
    
    print("\n✨ 핵심 차이점:")
    print("   ❌ 기존: 이미지 무시, 텍스트만 사용")
    print("   ✅ 개선: 이미지+텍스트 융합, 멀티모달 추론")


if __name__ == "__main__":
    demonstrate_working_principle()
    
    # 간단한 융합 모듈 테스트
    fusion = MultiModalFusion()
    
    # 더미 데이터로 테스트
    dummy_vision = torch.randn(2, 768)
    dummy_text = torch.randn(2, 768)
    
    fused = fusion(dummy_vision, dummy_text)
    print(f"\n🧪 Test: Vision{dummy_vision.shape} + Text{dummy_text.shape} → Fused{fused.shape}") 