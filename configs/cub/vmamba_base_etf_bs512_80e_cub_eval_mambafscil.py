_base_ = [
    '../_base_/models/vmamba_etf.py', '../_base_/datasets/cub_fscil.py',
    '../_base_/schedules/cub_80e.py', '../_base_/default_runtime.py'
]

# CUB requires different inc settings
inc_start = 100
inc_end = 200
inc_step = 10

# model settings
model = dict(backbone=dict(_delete_=True,
                           type='VMambaBackbone',
                           model_name='vmamba_tiny_s1l8',  # VMamba Tiny model name for MambaNeck
                           pretrained_path='./vssm1_tiny_0230s_ckpt_epoch_264.pth',  # VMamba Tiny pretrained weights
                           out_indices=(0, 1, 2, 3),  # Multi-scale features from all stages
                           frozen_stages=1,  # Freeze patch embedding and first stage
                           channel_first=True),
             neck=dict(type='MambaNeck',
                       version='ss2d',
                       in_channels=768,  # VMamba base stage4 channels
                       out_channels=768,
                       feat_size=7,
                       num_layers=2,
                       use_residual_proj=True,
                       use_new_branch=True,
                       detach_residual=False,
                       num_layers_new=3,
                       loss_weight_supp=100,
                       loss_weight_supp_novel=10,
                       loss_weight_sep=0.001,
                       loss_weight_sep_new=0.001,
                       param_avg_dim='0-1-3',
                       # Enhanced skip connection settings (MASC-M)
                       use_multi_scale_skip=True,
                       multi_scale_channels=[96, 192, 384]), 
             head=dict(type='ETFHead',
                       in_channels=768,
                       num_classes=200,
                       eval_classes=100,
                       loss=[
                           dict(type='DRLoss', loss_weight=1.0),
                           dict(type='CrossEntropyLoss', loss_weight=1.0)
                        ],
                       with_len=False),
             mixup=0.5,
             mixup_prob=0.5)

base_copy_list = (1, 1, 2, 2, 3, 3, 1, 1, 1, 1)
copy_list = (10, 10, 10, 10, 10, 10, 10, 10, 10, 10)
step_list = (200, 210, 220, 230, 240, 250, 260, 270, 280, 290)
finetune_lr = 0.05

# optimizer
optimizer = dict(type='SGD',
                 lr=finetune_lr,
                 momentum=0.9,
                 weight_decay=0.0005,
                 paramwise_cfg=dict(
                     custom_keys={
                         'neck.mlp_proj.': dict(lr_mult=0.2),
                         'neck.block.': dict(lr_mult=0.2),
                         'neck.residual_proj': dict(lr_mult=0.2),
                         'neck.pos_embed': dict(lr_mult=0.2),
                         'neck.pos_embed_new': dict(lr_mult=1),
                         # Enhanced skip connection components
                         'neck.multi_scale_adapters': dict(lr_mult=0.5),
                         'neck.skip_attention': dict(lr_mult=1.0),
                         'neck.skip_proj': dict(lr_mult=1.0),
                     }))

optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='CosineAnnealingCooldown',
    min_lr=None,
    min_lr_ratio=0.1,
    cool_down_ratio=0.1,
    cool_down_time=10,
    by_epoch=False,
    # warmup
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
    warmup_by_epoch=False) 

find_unused_parameters=True