import math
from typing import Sequence, Tuple, Type, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SwinUNETR import SwinUNETR
from model.Unet import UNet3D, UNet3D_baseline
from monai.networks.nets import SwinUNETR as m_SwinUNETR

# ==========================================
# 1. 跨模态注意力机制 (保持原有稳定逻辑)
# ==========================================
class CrossAttention3D(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor):
        B, C, D, H, W = query.shape
        query = query.flatten(2).transpose(1, 2)
        key_value = key_value.flatten(2).transpose(1, 2)
        N, M = query.shape[1], key_value.shape[1]
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(key_value).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(key_value).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        return x

class BidirectionalCrossAttention3D(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.ct_to_mr_attn = CrossAttention3D(dim, num_heads)
        self.mr_to_ct_attn = CrossAttention3D(dim, num_heads)
        self.fusion = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=1),
            nn.GroupNorm(8, dim),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, ct_feat: torch.Tensor, mr_feat: torch.Tensor):
        ct_enhanced = self.ct_to_mr_attn(ct_feat, mr_feat)
        mr_enhanced = self.mr_to_ct_attn(mr_feat, ct_feat)
        fused = self.fusion(torch.cat([ct_enhanced, mr_enhanced], dim=1))
        return ct_enhanced, mr_enhanced, fused

class GatedSkipFusion3D(nn.Module):
    """Fuse CT/MR skip features with a learnable channel-wise gate."""
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, ct_feat: torch.Tensor, mr_feat: torch.Tensor):
        ct_ctx = F.adaptive_avg_pool3d(ct_feat, output_size=1)
        mr_ctx = F.adaptive_avg_pool3d(mr_feat, output_size=1)
        alpha = self.gate(torch.cat([ct_ctx, mr_ctx], dim=1))
        return alpha * ct_feat + (1.0 - alpha) * mr_feat

# ==========================================
# 2. 核心动态模块：TDWB + SFM (含偏置专家与残差调制)
# ==========================================
class DynamicSemanticHead(nn.Module):
    def __init__(self, in_channels, out_channels, text_dim=256, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels

        # A. 基底专家：权重与偏置 (解决 Pred 全 0 的关键)
        self.weight_basis = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels, 3, 3, 3))
        self.bias_basis = nn.Parameter(torch.Tensor(num_experts, out_channels))
        
        # 初始化
        nn.init.kaiming_uniform_(self.weight_basis, a=math.sqrt(5))
        nn.init.zeros_(self.bias_basis)

        # B. 路由网络 (使用 LeakyReLU 预防梯度死区)
        self.router = nn.Sequential(
            nn.Linear(text_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, num_experts)
        )

        # C. 语义调制器 (使用残差设计)
        self.modulator = nn.Sequential(
            nn.Linear(text_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, out_channels),
            nn.Sigmoid() 
        )

    def forward(self, x, text_features):
        b = x.shape[0]
        
        # 计算路由与调制系数
        router_logits = self.router(text_features)
        routing_weights = F.softmax(router_logits, dim=-1)
        gamma = self.modulator(text_features).view(b, -1, 1, 1, 1)

        # 参数混合 (Weight & Bias Blending)
        combined_weight = torch.einsum('be,eout...->bout...', routing_weights, self.weight_basis)
        combined_bias = torch.einsum('be,eo->bo', routing_weights, self.bias_basis)

        # 高效并行的组卷积实现
        x_reshaped = x.view(1, -1, *x.shape[2:]) # [1, B*C_in, D, H, W]
        w_reshaped = combined_weight.view(-1, self.in_channels, 3, 3, 3) # [B*out_c, C_in, k...]
        b_reshaped = combined_bias.view(-1) # [B*out_c]
        
        out = F.conv3d(x_reshaped, w_reshaped, bias=b_reshaped, padding=1, groups=b)
        out = out.view(b, self.out_channels, *out.shape[2:]) 

        # 残差语义调制：1.0 + gamma 确保信号不被阉割
        out = out * (1.0 + gamma) 
        
        return out, router_logits, routing_weights

# ==========================================
# 3. 主模型类：MulModSeg
# ==========================================
class MulModSeg(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, backbone='swinunetr', encoding='word_embedding',
                 use_cross_attention=True, cross_attn_dim=None, cross_attn_heads=8, num_experts=2, case_text_alpha=0.2):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.case_text_alpha = case_text_alpha
        self.class_num = out_channels
        self.encoding = encoding
        self.num_experts = num_experts

        # 初始化骨干网络
        if backbone == 'swinunetr':
            self.backbone = SwinUNETR(img_size=img_size, in_channels=in_channels, out_channels=out_channels, feature_size=48)
            dec4_dim, decoder_out_dim = 768, 48
            self.skip_dims = {"enc0": 48, "enc1": 48, "enc2": 96, "enc3": 192, "hid3": 384}
        elif backbone == 'unet':
            self.backbone = UNet3D()
            dec4_dim, decoder_out_dim = 512, 64
            self.skip_dims = {"enc0": 64, "enc1": 128, "enc2": 256}
        else:
            raise NotImplementedError(f"Backbone {backbone} not supported.")
        
        # 跨模态融合逻辑
        if use_cross_attention:
            cross_attn_dim = cross_attn_dim or dec4_dim
            self.cross_attention = BidirectionalCrossAttention3D(dim=cross_attn_dim, num_heads=cross_attn_heads)
            self.cross_attn_proj = nn.Identity() if cross_attn_dim == dec4_dim else nn.Conv3d(cross_attn_dim, dec4_dim, 1)
            self.decoder_fusion = nn.Sequential(
                nn.Conv3d(decoder_out_dim * 2, decoder_out_dim, 1), 
                nn.LeakyReLU(0.1, inplace=True)
            )
            self.skip_fusion = nn.ModuleDict({k: GatedSkipFusion3D(v) for k, v in self.skip_dims.items()})

        # 文本特征对齐层
        if encoding == 'word_embedding':
            self.register_buffer('organ_embedding', torch.randn(out_channels, 512))
            self.text_to_vision = nn.Linear(512, 256)

        # 动态自适应头
        self.dynamic_head = DynamicSemanticHead(decoder_out_dim, out_channels, text_dim=256, num_experts=num_experts)

    def load_params(self, state_dict):
        """加载预训练权重到 backbone encoder（仅 SwinUNETR 有效）。
        train.py 传入的是已提取的 state_dict，load_from 需要包一层 dict。
        """
        if hasattr(self.backbone, 'load_from'):
            self.backbone.load_from({"state_dict": state_dict})
            print("[INFO] Loaded pretrained SwinUNETR encoder weights.")
        else:
            self.backbone.load_state_dict(state_dict, strict=False)
            print("[INFO] Loaded pretrained backbone weights (strict=False).")

    def _get_fused_text_feature(self, b, modality, case_text_embedding):
        # 修复 Shape 冲突：支持 [Modality, Class, Dim] 或 [Class, Dim]
        if self.organ_embedding.dim() == 3:
            midx = 0 if modality == 'MR' else 1
            static_tumor = F.normalize(self.organ_embedding[midx, 1], dim=-1)
        else:
            static_tumor = F.normalize(self.organ_embedding[1], dim=-1)

        if case_text_embedding is not None:
            case_tumor = F.normalize(case_text_embedding, dim=-1)
            fused = F.normalize(static_tumor.unsqueeze(0) + self.case_text_alpha * case_tumor, dim=-1)
        else:
            fused = static_tumor.unsqueeze(0).expand(b, -1)
        
        return F.leaky_relu(self.text_to_vision(fused), negative_slope=0.1)

    def _fuse_skip(self, name: str, ct_feat: Optional[torch.Tensor], mr_feat: Optional[torch.Tensor]):
        if ct_feat is None:
            return mr_feat
        if mr_feat is None:
            return ct_feat
        return self.skip_fusion[name](ct_feat, mr_feat)

    def forward(self, x_in, modality, x_in_mr=None, case_text_embedding=None):
        b = x_in.shape[0]
        
        # # 特征提取
        # if x_in_mr is not None and self.use_cross_attention:
        #     # 更激进：在 dec4 做 cross-attn 融合，然后用融合后的 dec4 只跑一次 decoder
        #     if hasattr(self.backbone, "forward_to_dec4") and hasattr(self.backbone, "forward_from_dec4"):
        #         enc0, enc1, enc2, enc3, dec4_ct, hid3 = self.backbone.forward_to_dec4(x_in, return_skips=True)
        #         dec4_mr = self.backbone.forward_to_dec4(x_in_mr, return_skips=False)
        #         _, _, fused_dec4 = self.cross_attention(dec4_ct, dec4_mr)
        #         out = self.backbone.forward_from_dec4(fused_dec4, enc0, enc1, enc2, enc3, hid3)
        #     else:
        #         # 兜底：若 backbone 不支持分段 forward，则退回到原来的 late fusion
        #         _, out_ct = self.backbone(x_in)
        #         _, out_mr = self.backbone(x_in_mr)
        #         out = self.decoder_fusion(torch.cat([out_ct, out_mr], dim=1))
        # else:
        #     _, out_ct = self.backbone(x_in)
        #     out = out_ct
        if x_in_mr is not None and self.use_cross_attention:
            # 1. CT/MR 都提取 dec4 + skip（Ablation 1：双路 skip 融合）
            ct_enc0, ct_enc1, ct_enc2, ct_enc3, dec4_ct, ct_hid3 = self.backbone.forward_to_dec4(x_in, return_skips=True)
            mr_enc0, mr_enc1, mr_enc2, mr_enc3, dec4_mr, mr_hid3 = self.backbone.forward_to_dec4(x_in_mr, return_skips=True)
            
            # 3. 跨模态注意力融合
            _, _, fused_dec4 = self.cross_attention(dec4_ct, dec4_mr)
            fused_dec4 = self.cross_attn_proj(fused_dec4)

            # 4. 浅层 skip 融合（逐层可学习门控）
            enc0 = self._fuse_skip("enc0", ct_enc0, mr_enc0)
            enc1 = self._fuse_skip("enc1", ct_enc1, mr_enc1)
            enc2 = self._fuse_skip("enc2", ct_enc2, mr_enc2)
            enc3 = self._fuse_skip("enc3", ct_enc3, mr_enc3) if "enc3" in self.skip_fusion else self._fuse_skip("enc2", ct_enc3, mr_enc3)
            hid3 = self._fuse_skip("hid3", ct_hid3, mr_hid3) if "hid3" in self.skip_fusion else self._fuse_skip("enc2", ct_hid3, mr_hid3)
            
            # 5. 单路解码：融合 dec4 + 融合 skips
            out = self.backbone.forward_from_dec4(fused_dec4, enc0, enc1, enc2, enc3, hid3)
        else:
            _, out = self.backbone(x_in)

        # 文本引导特征生成
        routing_feature = self._get_fused_text_feature(b, modality, case_text_embedding)
        
        # 动态头推理：返回 3 个值，完美对齐 train.py 293 行
        final_logits, router_logits, weights = self.dynamic_head(out, routing_feature)

        # 挂载属性供训练日志调用
        self.last_router_logits = router_logits 
        self.last_routing_weights = weights.detach().cpu()
        
        return final_logits, router_logits, weights

# ==========================================
# 4. 导出函数 (适配 train.py 导入)
# ==========================================
def UNet3D_cy(out_channels=1, act='relu'):
    model = UNet3D_baseline(n_class=out_channels, act=act)
    return model

def SwinUNETR_cy(out_channels, img_size=(96,96,96), in_channels=1, feature_size=48):
    return m_SwinUNETR(img_size=img_size, in_channels=in_channels, out_channels=out_channels, feature_size=feature_size)