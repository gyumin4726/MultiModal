import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import LOSSES

@LOSSES.register_module()
class PINNLoss(nn.Module):
    """Physics-Informed Neural Network style loss for FSCIL.
    
    This loss combines traditional classification loss with physics-inspired regularization
    terms to preserve feature space structure and class relationships.
    """
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 manifold_reg=0.01,
                 gradient_reg=0.01):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.manifold_reg = manifold_reg
        self.gradient_reg = gradient_reg

    def compute_manifold_regularization(self, feat):
        """Compute manifold regularization to preserve feature space structure."""
        # 특징 공간에서의 라플라시안 근사
        feat.requires_grad_(True)
        grad_feat = torch.autograd.grad(feat.sum(), feat, create_graph=True)[0]
        laplacian = torch.sum(torch.norm(grad_feat, p=2, dim=1))
        return laplacian

    def compute_gradient_preservation(self, feat, target):
        """Compute gradient preservation term to maintain class relationships."""
        # 클래스 간 그래디언트 보존
        cos_sim = F.cosine_similarity(feat.unsqueeze(1), feat.unsqueeze(0), dim=-1)
        target_sim = F.cosine_similarity(target.unsqueeze(1), target.unsqueeze(0), dim=-1)
        gradient_loss = F.mse_loss(cos_sim, target_sim)
        return gradient_loss

    def forward(self,
               feat,
               target,
               h_norm2=None,
               m_norm2=None,
               avg_factor=None):
        """Forward function.
        
        Args:
            feat (torch.Tensor): Feature vectors
            target (torch.Tensor): Target vectors
            h_norm2 (torch.Tensor, optional): Norm of feature vectors
            m_norm2 (torch.Tensor, optional): Norm of target vectors
            avg_factor (int, optional): Average factor used in loss calculation
        """
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)

        # 기본 DR Loss
        base_loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2))**2) / h_norm2)

        # PINN style 정규화 항들
        manifold_loss = self.compute_manifold_regularization(feat)
        gradient_loss = self.compute_gradient_preservation(feat, target)

        # 전체 손실 함수
        total_loss = (base_loss +
                     self.manifold_reg * manifold_loss +
                     self.gradient_reg * gradient_loss)

        return total_loss * self.loss_weight 