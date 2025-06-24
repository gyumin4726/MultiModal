import torch
import torch.nn as nn
import torch.nn.functional as F


class VQAMultiModalFusion(nn.Module):
    """VQA 특화 고급 멀티모달 융합 모듈"""
    
    def __init__(self, 
                 vision_dim=1024, 
                 text_dim=1024, 
                 hidden_dim=512, 
                 output_dim=1024,
                 num_heads=8):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # 1. 모달리티별 Feature Enhancement
        self.vision_enhancer = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.text_enhancer = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 2. Bi-directional Cross-Attention
        self.v2t_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.t2v_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 3. Adaptive Fusion Weights (학습 가능한 가중치)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # vision, text, interaction 가중치
            nn.Softmax(dim=-1)
        )
        
        # 4. Multi-scale Feature Fusion
        self.multi_scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // (2**i)),
                nn.ReLU(),
                nn.Linear(hidden_dim // (2**i), hidden_dim)
            ) for i in range(3)  # 3개의 스케일
        ])
        
        # 5. Final Output Layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 4, output_dim),  # enhanced + attended + gated + multi-scale
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        print(f"VQA MultiModal Fusion initialized: V{vision_dim}+T{text_dim} -> {output_dim}")
    
    def forward(self, vision_features, text_features):
        """
        Args:
            vision_features: (B, vision_dim) - 비전 피처
            text_features: (B, text_dim) - 텍스트 피처 (VQA 구조화된)
            
        Returns:
            fused_features: (B, output_dim) - 고급 융합된 피처
        """
        batch_size = vision_features.size(0)
        
        # 1. Feature Enhancement
        v_enhanced = self.vision_enhancer(vision_features)  # (B, hidden_dim)
        t_enhanced = self.text_enhancer(text_features)      # (B, hidden_dim)
        
        # 2. Sequence 차원 추가 (attention을 위해)
        v_seq = v_enhanced.unsqueeze(1)  # (B, 1, hidden_dim)
        t_seq = t_enhanced.unsqueeze(1)  # (B, 1, hidden_dim)
        
        # 3. Bi-directional Cross-Attention
        v2t_attended, v2t_weights = self.v2t_attention(v_seq, t_seq, t_seq)  # Vision queries Text
        t2v_attended, t2v_weights = self.t2v_attention(t_seq, v_seq, v_seq)  # Text queries Vision
        
        v2t_attended = v2t_attended.squeeze(1)  # (B, hidden_dim)
        t2v_attended = t2v_attended.squeeze(1)  # (B, hidden_dim)
        
        # 4. Adaptive Fusion Gating
        gate_input = torch.cat([v2t_attended, t2v_attended], dim=-1)  # (B, hidden_dim*2)
        fusion_weights = self.fusion_gate(gate_input)  # (B, 3)
        
        # 가중치 적용
        w_v, w_t, w_i = fusion_weights[:, 0:1], fusion_weights[:, 1:2], fusion_weights[:, 2:3]
        
        gated_features = (
            w_v * v_enhanced +           # 원본 vision
            w_t * t_enhanced +           # 원본 text  
            w_i * (v2t_attended + t2v_attended) / 2  # 상호작용
        )  # (B, hidden_dim)
        
        # 5. Multi-scale Feature Processing
        multi_scale_features = []
        for scale_layer in self.multi_scale_fusion:
            scale_feat = scale_layer(gated_features)
            multi_scale_features.append(scale_feat)
        
        multi_scale_combined = torch.stack(multi_scale_features, dim=1).mean(dim=1)  # (B, hidden_dim)
        
        # 6. Final Fusion
        final_input = torch.cat([
            v_enhanced,           # 강화된 vision
            t_enhanced,           # 강화된 text
            gated_features,       # 게이트된 융합
            multi_scale_combined  # 멀티스케일 융합
        ], dim=-1)  # (B, hidden_dim * 4)
        
        fused_features = self.output_layer(final_input)  # (B, output_dim)
        
        return fused_features


class HierarchicalVQAFusion(nn.Module):
    """계층적 VQA 융합 - 단계별 정보 융합"""
    
    def __init__(self, vision_dim=1024, text_dim=1024, output_dim=1024):
        super().__init__()
        
        # Level 1: Low-level feature alignment
        self.level1_fusion = VQAMultiModalFusion(
            vision_dim=vision_dim,
            text_dim=text_dim,
            hidden_dim=256,
            output_dim=512,
            num_heads=4
        )
        
        # Level 2: High-level semantic fusion
        self.level2_fusion = VQAMultiModalFusion(
            vision_dim=512,  # from level 1
            text_dim=text_dim,
            hidden_dim=512,
            output_dim=output_dim,
            num_heads=8
        )
        
        # Residual connection
        self.residual_proj = nn.Linear(vision_dim + text_dim, output_dim)
        
    def forward(self, vision_features, text_features):
        """계층적 융합"""
        
        # Level 1: 기본 융합
        level1_fused = self.level1_fusion(vision_features, text_features)  # (B, 512)
        
        # Level 2: 고급 융합 (Level 1 결과 + 원본 텍스트)
        level2_fused = self.level2_fusion(level1_fused, text_features)  # (B, output_dim)
        
        # Residual connection
        residual = self.residual_proj(torch.cat([vision_features, text_features], dim=-1))
        
        final_output = level2_fused + residual  # Skip connection
        
        return final_output


class EnhancedMultiModalProcessor:
    """향상된 멀티모달 처리기 - VQA 특화"""
    
    def __init__(self, vision_encoder, text_encoder, language_model, fusion_type='hierarchical'):
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder  
        self.language_model = language_model
        self.fusion_type = fusion_type
        
        # 융합 모듈 선택
        if fusion_type == 'hierarchical':
            self.fusion_model = HierarchicalVQAFusion(
                vision_dim=1024,
                text_dim=1024,
                output_dim=1024
            )
        else:  # 'advanced'
            self.fusion_model = VQAMultiModalFusion(
                vision_dim=1024,
                text_dim=1024,
                hidden_dim=512,
                output_dim=1024
            )
        
        if torch.cuda.is_available():
            self.fusion_model = self.fusion_model.cuda()
            
        print(f"✅ Enhanced MultiModal Processor initialized with {fusion_type} fusion")
            
    def process_multimodal_input(self, image_tensor, question_text, choices):
        """VQA 특화 멀티모달 입력 처리"""
        
        # 1. Vision 피처 추출
        with torch.no_grad():
            vision_features = self.vision_encoder(image_tensor)  # (1, 1024)
        
        # 2. VQA 구조화 텍스트 인코딩
        text_features, qc_attention = self.text_encoder(question_text, choices)  # (1, 1024)
        
        # 3. 고급 멀티모달 융합
        fused_features = self.fusion_model(vision_features, text_features)  # (1, 1024)
        
        # 4. 언어 모델로 VQA 추론
        answer = self.language_model.answer_question_with_features(
            question_text, 
            choices, 
            vision_features=vision_features,
            text_features=text_features,
            fused_features=fused_features
        )
        
        return answer, {
            'vision_features': vision_features,
            'text_features': text_features, 
            'fused_features': fused_features,
            'qc_attention': qc_attention
        }


if __name__ == "__main__":
    print("🚀 Enhanced VQA MultiModal System:")
    print("="*60)
    
    print("\n1️⃣ Vision Processing:")
    print("   📸 Image → ViT-Large → MASC-V → 1024D vision features")
    
    print("\n2️⃣ VQA Text Processing:")
    print("   📝 Question + Choices → MPNet → Question-Choice Attention → 1024D text features")
    
    print("\n3️⃣ Hierarchical Fusion:")
    print("   🔗 Level 1: Low-level alignment (Vision ↔ Text)")
    print("   🔗 Level 2: High-level semantic fusion + Residual")
    print("   🔗 Result: 1024D fused features")
    
    print("\n4️⃣ Advanced Reasoning:")
    print("   🧠 Fused features → Question-type specific prompts → Self-verification → Answer")
    
    print("\n✨ 주요 개선사항:")
    print("   🔥 Question-Choice 구조화 인코딩")
    print("   🔥 Bi-directional Cross-Attention")
    print("   🔥 Adaptive Fusion Gating")
    print("   🔥 Multi-scale Feature Processing")
    print("   🔥 Hierarchical Fusion Architecture")
    print("   🔥 VQA-specific Prompt Engineering")
    
    # 고급 융합 모듈 테스트
    fusion = VQAMultiModalFusion(vision_dim=1024, text_dim=1024, output_dim=1024)
    
    # 더미 데이터로 테스트
    dummy_vision = torch.randn(2, 1024)
    dummy_text = torch.randn(2, 1024)
    
    fused = fusion(dummy_vision, dummy_text)
    print(f"\n🧪 Test: Vision{dummy_vision.shape} + Text{dummy_text.shape} → Fused{fused.shape}")
    
    # 계층적 융합 테스트
    hierarchical_fusion = HierarchicalVQAFusion(vision_dim=1024, text_dim=1024, output_dim=1024)
    hierarchical_fused = hierarchical_fusion(dummy_vision, dummy_text)
    print(f"🧪 Hierarchical: Vision{dummy_vision.shape} + Text{dummy_text.shape} → Fused{hierarchical_fused.shape}") 