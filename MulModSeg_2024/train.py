import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import json
import argparse
import time
import random

import warnings
warnings.filterwarnings("ignore")

# Add project root to sys.path so dataloaders are importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from copy import deepcopy
from contextlib import contextmanager

from tensorboardX import SummaryWriter

from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.networks.nets import UNet

from model.MulModSeg import MulModSeg, UNet3D_cy, SwinUNETR_cy
from dataloader_data1 import get_loader_data1
from dataloader_bone_tumor import get_loader_bone_tumor

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
# from monai.inferers import sliding_window_inference_cy  # Not used, commented out
from itertools import cycle

# Custom modules for experiment framework
from utils.custom_losses import get_loss_function, boundary_dice_loss
from utils.enhanced_validation import enhanced_validation
from utils.case_text_embedding import CaseTextEmbeddingStore, get_case_text_embedding_from_batch
from utils.pretrained_encoder import load_pretrained_encoder, freeze_encoder

# ================= EMA / SWA helpers =================
@torch.no_grad()
def _ema_update_(ema_model, model, decay: float):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    # keep buffers in sync (e.g. norm stats)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


@contextmanager
def _temporary_state_dict(model: torch.nn.Module, state_dict: dict):
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state_dict, strict=False)
    try:
        yield
    finally:
        model.load_state_dict(backup, strict=False)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.multiprocessing.set_sharing_strategy('file_system')

def plot_routing_trends(epoch_list, expert0_weights, expert1_weights, save_path):
    """
    绘制 MoE 路由权重随 Epoch 的变化趋势图
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, expert0_weights, label='Expert 0 (e.g., Pelvis)', marker='o', color='blue', linewidth=2)
    plt.plot(epoch_list, expert1_weights, label='Expert 1 (e.g., Femur)', marker='s', color='orange', linewidth=2)
    
    plt.title('MoE Routing Weights Trend Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Routing Weight', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Assignment (0.5)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def compute_custom_seg_loss(args, loss_function, logit_map, label):
    """
    自定义 loss 统一接收 logits，避免在训练入口和 loss 内部重复 softmax。
    """
    y_onehot = F.one_hot(
        label[:, 0, :, :, :].long(),
        num_classes=args.num_class,
    ).permute(0, 4, 1, 2, 3).float()
    loss = loss_function(logit_map, y_onehot)
    return loss, torch.tensor(0.0, device=logit_map.device)


def get_case_text_embedding(args, batch, modality=None):
    if not getattr(args, "use_case_text_embedding", False):
        return None
    store = getattr(args, "case_text_store", None)
    if store is None:
        return None
    return get_case_text_embedding_from_batch(batch, store, args.device, modality=modality)

# def train_mix(args, train_loader_ct, train_loader_mr, model, optimizer, loss_function, loss_seg_CE):
#     model.train()
#     loss_bce_ave = 0
#     loss_dice_ave = 0
    
#     # Track Routing Weights
#     route_w_ave = [0.0, 0.0]
#     route_count = 0

#     use_cross_attention = getattr(args, 'use_cross_attention', False)

#     if use_cross_attention and hasattr(model, 'use_cross_attention') and model.use_cross_attention:
#         if train_loader_mr is None:
#             # 配对 loader 模式（get_loader_paired_bone_tumor）：
#             # 每个 batch 含 'ct', 'mr', 'label', 'name'，CT 和 MR 来自同一病例
#             _len = len(train_loader_ct)
#             for step, batch in enumerate(train_loader_ct):
#                 x_ct = batch['ct'].to(args.device)
#                 x_mr = batch['mr'].to(args.device)
#                 y = batch['label'].float().to(args.device)
#                 case_text_embedding = get_case_text_embedding(args, batch, modality='CT')

#                 logit_map = model(x_ct, 'CT', x_in_mr=x_mr, case_text_embedding=case_text_embedding)

#                 if args.loss_type == 'dicece':
#                     term_seg_Dice = loss_function.forward(logit_map, y)
#                     term_seg_BCE = loss_seg_CE.forward(logit_map, y)
#                     loss = term_seg_Dice + term_seg_BCE
#                 else:
#                     loss, term_seg_BCE = compute_custom_seg_loss(args, loss_function, logit_map, y)
#                     term_seg_Dice = loss

#                 # ================= MoE Entropy Loss & Tracking =================
#                 entropy_loss_val = 0.0
#                 route_str = "N/A"
#                 if hasattr(model, 'last_routing_entropy_loss') and model.last_routing_entropy_loss is not None:
#                     entropy_loss_val = model.last_routing_entropy_loss.item()
#                     loss = loss + args.entropy_weight * model.last_routing_entropy_loss
                
#                 if hasattr(model, 'last_routing_weights') and model.last_routing_weights is not None:
#                     batch_route_w = model.last_routing_weights.mean(dim=0).tolist()
#                     if len(batch_route_w) >= 2:
#                         route_str = f"[{batch_route_w[0]:.2f}, {batch_route_w[1]:.2f}]"
#                         route_w_ave[0] += batch_route_w[0]
#                         route_w_ave[1] += batch_route_w[1]
#                         route_count += 1
#                 # ===============================================================

#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()

#                 if step % 10 == 0:
#                     if args.loss_type == 'dicece':
#                         print(
#                             "Epoch=%d: Training (%d / %d Steps) (dice=%2.5f, bce=%2.5f, route_ent=%2.5f, weights=%s)" % (
#                                 args.epoch, step, _len, term_seg_Dice.item(), term_seg_BCE.item(), entropy_loss_val, route_str)
#                         )
#                     else:
#                         print(
#                             "Epoch=%d: Training (%d / %d Steps) (loss=%2.5f, route_ent=%2.5f, weights=%s)" % (
#                                 args.epoch, step, _len, loss.item(), entropy_loss_val, route_str)
#                         )
#                 if args.loss_type == 'dicece':
#                     loss_bce_ave += term_seg_BCE.item()
#                 loss_dice_ave += loss.item()
#         else:
#             # 分离 loader 模式：zip(ct_loader, mr_loader)，每个 batch 含 'image', 'label', 'modality'
#             _len = min(len(train_loader_ct), len(train_loader_mr))
#             for step, (batch_ct, batch_mr) in enumerate(zip(train_loader_ct, train_loader_mr)):
#                 x_ct = batch_ct["image"].to(args.device)
#                 y_ct = batch_ct["label"].float().to(args.device)
#                 z_ct = batch_ct['modality']
#                 x_mr = batch_mr["image"].to(args.device)
#                 case_text_embedding = get_case_text_embedding(args, batch_ct, modality=z_ct[0])

#                 min_batch = min(x_ct.shape[0], x_mr.shape[0])
#                 x_ct = x_ct[:min_batch]
#                 y_ct = y_ct[:min_batch]
#                 x_mr = x_mr[:min_batch]
#                 if case_text_embedding is not None:
#                     case_text_embedding = case_text_embedding[:min_batch]

#                 logit_map = model(x_ct, z_ct[0], x_in_mr=x_mr, case_text_embedding=case_text_embedding)

#                 if args.loss_type == 'dicece':
#                     term_seg_Dice = loss_function.forward(logit_map, y_ct)
#                     term_seg_BCE = loss_seg_CE.forward(logit_map, y_ct)
#                     loss = term_seg_Dice + term_seg_BCE
#                 else:
#                     loss, term_seg_BCE = compute_custom_seg_loss(args, loss_function, logit_map, y_ct)
#                     term_seg_Dice = loss

#                 # ================= MoE Entropy Loss & Tracking =================
#                 entropy_loss_val = 0.0
#                 route_str = "N/A"
#                 if hasattr(model, 'last_routing_entropy_loss') and model.last_routing_entropy_loss is not None:
#                     entropy_loss_val = model.last_routing_entropy_loss.item()
#                     loss = loss + args.entropy_weight * model.last_routing_entropy_loss
                
#                 if hasattr(model, 'last_routing_weights') and model.last_routing_weights is not None:
#                     batch_route_w = model.last_routing_weights.mean(dim=0).tolist()
#                     if len(batch_route_w) >= 2:
#                         route_str = f"[{batch_route_w[0]:.2f}, {batch_route_w[1]:.2f}]"
#                         route_w_ave[0] += batch_route_w[0]
#                         route_w_ave[1] += batch_route_w[1]
#                         route_count += 1
#                 # ===============================================================

#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()

#                 if step % 10 == 0:
#                     if args.loss_type == 'dicece':
#                         print(
#                             "Epoch=%d: Training (%d / %d Steps) (dice=%2.5f, bce=%2.5f, route_ent=%2.5f, weights=%s)" % (
#                                 args.epoch, step, _len, term_seg_Dice.item(), term_seg_BCE.item(), entropy_loss_val, route_str)
#                         )
#                     else:
#                         print(
#                             "Epoch=%d: Training (%d / %d Steps) (loss=%2.5f, route_ent=%2.5f, weights=%s)" % (
#                                 args.epoch, step, _len, loss.item(), entropy_loss_val, route_str)
#                         )
#                 if args.loss_type == 'dicece':
#                     loss_bce_ave += term_seg_BCE.item()
#                 loss_dice_ave += loss.item()
#     else:
#         # 原有模式：分别处理CT和MR，每个 batch 含 'image', 'label', 'modality'
#         _len = min(len(train_loader_ct), len(train_loader_mr))
#         for step, (batch_ct, batch_mr) in enumerate(zip(train_loader_ct, train_loader_mr)):
#             for batch in [batch_ct, batch_mr]:
#                 x, y, z, name = batch["image"].to(args.device), batch["label"].float().to(args.device), batch['modality'], batch['name']
#                 case_text_embedding = get_case_text_embedding(args, batch, modality=z[0])
#                 if args.with_text_embedding == 1:
#                     logit_map = model(x, z[0], case_text_embedding=case_text_embedding)
#                 else:
#                     logit_map = model(x)

#                 if args.loss_type == 'dicece':
#                     term_seg_Dice = loss_function.forward(logit_map, y)
#                     term_seg_BCE = loss_seg_CE.forward(logit_map, y)
#                     loss = term_seg_Dice + term_seg_BCE
#                 else:
#                     loss, term_seg_BCE = compute_custom_seg_loss(args, loss_function, logit_map, y)
#                     term_seg_Dice = loss

#                 # ================= MoE Entropy Loss & Tracking =================
#                 entropy_loss_val = 0.0
#                 route_str = "N/A"
#                 if hasattr(model, 'last_routing_entropy_loss') and model.last_routing_entropy_loss is not None:
#                     entropy_loss_val = model.last_routing_entropy_loss.item()
#                     loss = loss + args.entropy_weight * model.last_routing_entropy_loss
                
#                 if hasattr(model, 'last_routing_weights') and model.last_routing_weights is not None:
#                     batch_route_w = model.last_routing_weights.mean(dim=0).tolist()
#                     if len(batch_route_w) >= 2:
#                         route_str = f"[{batch_route_w[0]:.2f}, {batch_route_w[1]:.2f}]"
#                         route_w_ave[0] += batch_route_w[0]
#                         route_w_ave[1] += batch_route_w[1]
#                         route_count += 1
#                 # ===============================================================

#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()

#                 if step % 10 == 0:
#                     if args.loss_type == 'dicece':
#                         print(
#                             "Epoch=%d: Training (%d / %d Steps) (dice=%2.5f, bce=%2.5f, route_ent=%2.5f, weights=%s)" % (
#                                 args.epoch, step, _len, term_seg_Dice.item(), term_seg_BCE.item(), entropy_loss_val, route_str)
#                         )
#                     else:
#                         print(
#                             "Epoch=%d: Training (%d / %d Steps) (loss=%2.5f, route_ent=%2.5f, weights=%s)" % (
#                                 args.epoch, step, _len, loss.item(), entropy_loss_val, route_str)
#                         )
#                 if args.loss_type == 'dicece':
#                     loss_bce_ave += term_seg_BCE.item()
#                 loss_dice_ave += loss.item()

#     if args.loss_type == 'dicece':
#         print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f' % (args.epoch, loss_dice_ave/_len, loss_bce_ave/_len))
#     else:
#         print('Epoch=%d: ave_loss=%2.5f' % (args.epoch, loss_dice_ave/_len))

#     avg_w0 = route_w_ave[0] / max(1, route_count)
#     avg_w1 = route_w_ave[1] / max(1, route_count)
#     return loss_dice_ave/_len, loss_bce_ave/_len if args.loss_type == 'dicece' else 0.0, avg_w0, avg_w1

def train_mix(args, train_loader_ct, train_loader_mr, model, optimizer, loss_function, loss_seg_CE):
    model.train()
    loss_dice_ave = 0
    loss_bce_ave = 0
    loss_routing_ave = 0 #路由损失的累加器
    route_w_ave = [0.0, 0.0]
    route_count = 0

    # 这里的 _len 逻辑需要根据实际使用的 loader 确定
    _len = len(train_loader_ct) 
    
    for step, batch in enumerate(train_loader_ct):
        term_boundary = None
        # 1. 数据准备
        x_ct = batch['ct'].to(args.device)
        x_mr = batch['mr'].to(args.device) if 'mr' in batch else None
        y = batch['label'].float().to(args.device)
        names = batch['name'] # 用于生成强监督标签
        
        case_text_embedding = get_case_text_embedding(args, batch, modality='CT')

        # 2. 前向传播：注意这里现在接收 3 个返回值
        # logit_map: 分割结果, router_logits: 路由原始输出, routing_weights: 权重(alpha, beta)
        logit_map, router_logits, routing_weights = model(
            x_ct, 'CT', x_in_mr=x_mr, case_text_embedding=case_text_embedding
        )

        # 3. 计算分割 Loss (保持原有逻辑)
        if args.loss_type == 'dicece':
            term_seg_Dice = loss_function.forward(logit_map, y)
            term_seg_BCE = loss_seg_CE.forward(logit_map, y)
            loss = term_seg_Dice + term_seg_BCE
        else:
            loss, term_seg_BCE = compute_custom_seg_loss(args, loss_function, logit_map, y)
            term_seg_Dice = loss

        # 3b. 边界带 Dice（任务一）：仅在 GT 肿瘤轮廓邻域施力，强调边缘对齐
        term_boundary = None
        bw = float(getattr(args, "boundary_loss_weight", 0.0) or 0.0)
        if bw > 0.0:
            y_cls = (y[:, 0] > 0.5).long()
            y_onehot = F.one_hot(y_cls, num_classes=args.num_class).permute(0, 4, 1, 2, 3).float()
            term_boundary = boundary_dice_loss(
                logit_map,
                y_onehot,
                kernel_size=int(getattr(args, "boundary_kernel", 3)),
            )
            loss = loss + bw * term_boundary

        # 4. ================= 核心：强监督 MoE 路由 Loss =================
        # 动态构建 ID 到批次的映射表 (只在第一步读取，不影响速度)
        if step == 0 and not hasattr(args, "id_to_batch"):
            # 请确保这里的路径能正确找到你的 split_seed42.json
            json_path = os.path.join('/root/autodl-tmp/segmentation-test-main/splits/split_seed42.json') 
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    split_data = json.load(f)
                
                args.id_to_batch = {}
                # 把 train 和 val 的所有样本合并成一个查阅字典
                all_samples = split_data.get('train', []) + split_data.get('val', [])
                for item in all_samples:
                    # item 的格式是 "第X批/123456"
                    parts = item.split('/')
                    if len(parts) == 2:
                        batch_name = parts[0] # "第3批"
                        case_id = parts[1]    # "11461243"
                        args.id_to_batch[case_id] = batch_name
                print(f"[INFO] 成功加载批次映射表，共 {len(args.id_to_batch)} 个样本。")
            except Exception as e:
                print(f"[ERROR] 读取 JSON 失败: {e}。将全部默认分配给专家 1。")
                args.id_to_batch = {} # 兜底

        # 根据 batch_name 自动判断：第4、5批为骨盆(Expert 0)，其余为股骨(Expert 1)
        route_targets = []
            
        for n in names:
            # 提取纯数字 ID (兼容 names[0] 可能带路径或后缀的情况)
            # 例如: '/path/to/11023143.nii.gz' -> '11023143'
            n_str = str(n)
            case_id = os.path.basename(n_str).split('.')[0] 

            # 去字典里查它的批次，查不到默认是 ""
            original_batch = args.id_to_batch.get(case_id, "") 
            
            # 查到了就精准分配
            if '第4批' in original_batch or '第5批' in original_batch:
                route_targets.append(0) # 骨盆专家
            else:
                route_targets.append(1) # 股骨专家

        route_targets = torch.tensor(route_targets, dtype=torch.long, device=args.device)
        
        # --- 核心验证：检查当前 Batch 是否存在骨盆样本 ---
        num_pelvis = (route_targets == 0).sum().item()

        # 计算路由交叉熵
        loss_routing = F.cross_entropy(router_logits, route_targets)
        loss = loss + 0.3 * loss_routing
        # ===============================================================

        # 5. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update (step-level)
        if getattr(args, "ema_model", None) is not None and getattr(args, "use_ema", False):
            _ema_update_(args.ema_model, model, decay=float(getattr(args, "ema_decay", 0.999)))

        # 6. 统计与记录
        batch_w = routing_weights.mean(dim=0).detach().cpu().tolist()
        route_w_ave[0] += batch_w[0]
        route_w_ave[1] += batch_w[1]
        route_count += 1

        if step % 10 == 0:
            b_str = ""
            if term_boundary is not None:
                b_str = f" Bdry={term_boundary.item():.4f}"
            print(f"Epoch={args.epoch} [{step}/{_len}] Dice={term_seg_Dice.item():.4f}{b_str} "
                  f"Route_CE={loss_routing.item():.4f} Weights=[{batch_w[0]:.2f}, {batch_w[1]:.2f}]")

        loss_dice_ave += term_seg_Dice.item()
        loss_bce_ave += term_seg_BCE.item()
        loss_routing_ave += loss_routing.item() # <--- [新增2] 累加单步的路由损失

    avg_w0 = route_w_ave[0] / max(1, route_count)
    avg_w1 = route_w_ave[1] / max(1, route_count)
    avg_route_ce = loss_routing_ave / max(1, route_count) # <--- [新增3] 计算平均路由损失
    return loss_dice_ave/_len, loss_bce_ave/_len, avg_w0, avg_w1, avg_route_ce

def train(args, train_loader, model, optimizer, loss_function, loss_seg_CE=None):
    model.train()
    loss_total_ave = 0
    loss_dice_ave = 0
    loss_bce_ave = 0
    route_w_ave = [0.0, 0.0]
    route_count = 0

    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y, z, name = batch["image"].to(args.device), batch["label"].float().to(args.device), batch['modality'], batch['name']
        case_text_embedding = get_case_text_embedding(args, batch, modality=z[0])
        if args.with_text_embedding == 1:
            output = model(x, z[0], case_text_embedding=case_text_embedding)
        else:
            output = model(x)
        # MulModSeg returns (logits, router_logits, routing_weights); unpack
        logit_map = output[0] if isinstance(output, tuple) else output

        # Compute loss based on loss type
        if args.loss_type == 'dicece':
            term_seg_Dice = loss_function.forward(logit_map, y)
            term_seg_BCE = loss_seg_CE.forward(logit_map, y)
            loss = term_seg_Dice + term_seg_BCE
            loss_dice_ave += term_seg_Dice.item()
            loss_bce_ave += term_seg_BCE.item()
        else:
            loss, term_seg_BCE = compute_custom_seg_loss(args, loss_function, logit_map, y)
            term_seg_Dice = loss  # For logging
            loss_dice_ave += loss.item()

        # ================= MoE Entropy Loss & Tracking =================
        entropy_loss_val = 0.0
        route_str = "N/A"
        if hasattr(model, 'last_routing_entropy_loss') and model.last_routing_entropy_loss is not None:
            entropy_loss_val = model.last_routing_entropy_loss.item()
            loss = loss + args.entropy_weight * model.last_routing_entropy_loss
        
        if hasattr(model, 'last_routing_weights') and model.last_routing_weights is not None:
            batch_route_w = model.last_routing_weights.mean(dim=0).tolist()
            if len(batch_route_w) >= 2:
                route_str = f"[{batch_route_w[0]:.2f}, {batch_route_w[1]:.2f}]"
                route_w_ave[0] += batch_route_w[0]
                route_w_ave[1] += batch_route_w[1]
                route_count += 1
        # ===============================================================

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if args.loss_type == 'dicece':
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) (dice=%2.5f, bce=%2.5f, route_ent=%2.5f, w=%s)" % (
                    args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item(), entropy_loss_val, route_str)
            )
        else:
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) (%s_loss=%2.5f, route_ent=%2.5f, w=%s)" % (
                    args.epoch, step, len(train_loader), args.loss_type, loss.item(), entropy_loss_val, route_str)
            )

        loss_total_ave += loss.item()

    if args.loss_type == 'dicece':
        print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f' % (args.epoch, loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)))
    else:
        print('Epoch=%d: ave_%s_loss=%2.5f' % (args.epoch, args.loss_type, loss_dice_ave/len(epoch_iterator)))

    avg_w0 = route_w_ave[0] / max(1, route_count)
    avg_w1 = route_w_ave[1] / max(1, route_count)
    return loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator), avg_w0, avg_w1

def process(args):
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to: {args.seed}")

    args.device = torch.device(f"cuda:{args.device}") 
    torch.cuda.set_device(f"{args.device}")  

    if getattr(args, "resume", None) and getattr(args, "finetune_from", None):
        raise ValueError("--resume and --finetune_from are mutually exclusive. Use --resume to continue training, or --finetune_from to restart with reset LR.")

    # prepare the 3D model
    if args.with_text_embedding == 1:
        use_cross_attention = getattr(args, 'use_cross_attention', False)
        cross_attn_heads = getattr(args, 'cross_attn_heads', 8)
        model = MulModSeg(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=args.num_class,
                    backbone=args.backbone,
                    encoding=args.trans_encoding,
                    use_cross_attention=use_cross_attention,
                    cross_attn_heads=cross_attn_heads,
                    case_text_alpha=args.case_text_alpha,
                    )
        if use_cross_attention:
            print(f"[INFO] Cross-attention enabled with {cross_attn_heads} heads")
    else:
        if args.backbone == 'unet':
            model = UNet3D_cy(
            out_channels=args.num_class
            )
        elif args.backbone == 'swinunetr':
            model = SwinUNETR_cy(
                img_size=(args.roi_x, args.roi_y, args.roi_z),
                out_channels=args.num_class
            )

    #Load pre-trained weights
    if args.pretrain is not None:
        model.load_params(torch.load(args.pretrain)["state_dict"])

    # Load encoder-only pretrained weights (SSL or BTCV/MONAI)
    if getattr(args, "pretrain_encoder_only", None):
        loaded, missing, unexpected = load_pretrained_encoder(
            model, args.pretrain_encoder_only, strict=False, verbose=True
        )
        print(f"[INFO] pretrain_encoder_only: loaded={loaded}, missing={missing}, unexpected={unexpected}")

    # Apply encoder freeze strategy
    if getattr(args, "freeze_level", "none") != "none":
        if getattr(args, "pretrain_encoder_only", None) is None and args.pretrain is None:
            print("[WARNING] --freeze_level set but no pretrained encoder loaded; freezing random weights.")
        freeze_encoder(model, args.freeze_level)

    if args.with_text_embedding == 1 and args.trans_encoding == 'word_embedding':
        word_embedding = torch.load(args.word_embedding, map_location=args.device)
        if isinstance(word_embedding, dict):
            for key in ['organ_embedding', 'embedding', 'embeddings', 'weight']:
                if key in word_embedding and isinstance(word_embedding[key], torch.Tensor):
                    word_embedding = word_embedding[key]
                    break
            else:
                tensor_vals = [v for v in word_embedding.values() if isinstance(v, torch.Tensor)]
                if not tensor_vals:
                    raise TypeError(f"Loaded word_embedding from {args.word_embedding} is a dict without any Tensor values.")
                word_embedding = tensor_vals[0]
        if not isinstance(word_embedding, torch.Tensor):
            raise TypeError(f"Loaded word_embedding from {args.word_embedding} must be a Tensor or dict of Tensors, got {type(word_embedding)}.")
        model.organ_embedding.data = word_embedding.float()
        print('load word embedding')

    args.case_text_store = None
    if (
        args.with_text_embedding == 1
        and args.trans_encoding == 'word_embedding'
        and args.use_case_text_embedding
    ):
        if not args.case_text_embedding:
            raise ValueError("--use_case_text_embedding requires --case_text_embedding.")
        args.case_text_store = CaseTextEmbeddingStore(args.case_text_embedding)
        print(f"[INFO] Loaded case text embeddings from: {args.case_text_embedding}")

    model.to(args.device)
    model.train()

    # Optional EMA/SWA models (evaluation/saving only)
    args.ema_model = None
    args.swa_model = None
    if getattr(args, "use_ema", False):
        args.ema_model = deepcopy(model).to(args.device)
        args.ema_model.eval()
        for p in args.ema_model.parameters():
            p.requires_grad_(False)
        print(f"[INFO] EMA enabled (decay={args.ema_decay})")

    if getattr(args, "use_swa", False):
        from torch.optim.swa_utils import AveragedModel
        args.swa_model = AveragedModel(model).to(args.device)
        args.swa_model.eval()
        for p in args.swa_model.parameters():
            p.requires_grad_(False)
        print(f"[INFO] SWA enabled (start_epoch={args.swa_start}, update_every={args.swa_update_every})")

    # Finetune: load model weights only, reset optimizer/scheduler/epoch
    if getattr(args, "finetune_from", None):
        ckpt = torch.load(args.finetune_from, map_location=args.device)
        state_dict = ckpt.get("net", None)
        if state_dict is None:
            state_dict = ckpt.get("state_dict", None)
        if state_dict is None and isinstance(ckpt, dict):
            # allow passing a raw state_dict
            state_dict = ckpt
        if state_dict is None or not isinstance(state_dict, dict):
            raise ValueError(f"Unrecognized checkpoint format in {args.finetune_from}. Expected keys: 'net' or 'state_dict', or a raw state_dict dict.")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Finetune: loaded weights from {args.finetune_from} (strict=False). Missing={len(missing)}, Unexpected={len(unexpected)}")
        # restart training from scratch unless user explicitly sets --epoch
        try:
            args.epoch = int(getattr(args, "epoch", 0))
        except Exception:
            args.epoch = 0

        # If EMA/SWA enabled, re-init them from the loaded weights
        if getattr(args, "ema_model", None) is not None:
            args.ema_model.load_state_dict(model.state_dict(), strict=False)
        if getattr(args, "swa_model", None) is not None:
            args.swa_model = args.swa_model.__class__(model).to(args.device)
            args.swa_model.eval()
            for p in args.swa_model.parameters():
                p.requires_grad_(False)

    # Initialize loss functions based on loss_type
    if args.loss_type == 'dicece':
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=0.0)
        loss_seg_CE = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0.0, lambda_ce=1.0)
        print(f"[INFO] Using loss function: DiceCE")
    else:
        loss_function = get_loss_function(
            loss_type=args.loss_type,
            alpha=args.loss_alpha,
            beta=args.loss_beta,
            gamma=args.loss_gamma,
            num_classes=args.num_class
        )
        loss_seg_CE = None
        print(f"[INFO] Using loss function: {args.loss_type}")
        print(f"[INFO] Loss parameters: alpha={args.loss_alpha}, beta={args.loss_beta}, gamma={args.loss_gamma}")
    if args.backbone == 'unetpp':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, 
                                    nesterov=False, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if not getattr(args, "fixed_lr", False):
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=args.warmup_epoch,
            max_epochs=args.max_epoch,
            eta_min=1e-5,
        )
    else:
        print(f"[INFO] Fixed LR mode enabled: lr={optimizer.param_groups[0]['lr']}")

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch']
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print('success resume from ', args.resume)

    torch.backends.cudnn.benchmark = True

    # train loader
    if args.dataset == 'data1':
        if args.train_modality == 'MIX':
            train_loader_ct = get_loader_data1(train_modality='CT', phase='train', persistent=True)
            train_loader_mr = get_loader_data1(train_modality='MR', phase='train', persistent=True)
        else:
            train_loader = get_loader_data1(train_modality=args.train_modality, phase='train', persistent=True)
    elif args.dataset == 'bone_tumor':
        roi_size = (args.roi_x, args.roi_y, args.roi_z)
        if args.train_modality == 'MIX':
            use_cross_attention = getattr(args, 'use_cross_attention', False)
            if use_cross_attention:
                from dataloader_bone_tumor import get_loader_paired_bone_tumor
                train_loader_ct = get_loader_paired_bone_tumor(
                    root_dir=args.data_root_path,
                    phase='train',
                    batch_size=args.batch_size,
                    roi_size=roi_size,
                    train_num_samples=args.num_samples,
                    persistent=True,
                    num_workers=args.num_workers,
                    random_seed=args.seed,
                )
                train_loader_mr = None
                val_loader = get_loader_paired_bone_tumor(
                    root_dir=args.data_root_path,
                    phase='val',
                    batch_size=1,
                    roi_size=roi_size,
                    persistent=True,
                    num_workers=0,
                    random_seed=args.seed,
                )
                print(f"[INFO] Cross-attention mode: Using paired CT+MR loader")
            else:
                train_loader = get_loader_bone_tumor(
                    root_dir=args.data_root_path,
                    modality='MIX',
                    phase='train',
                    batch_size=args.batch_size,
                    roi_size=roi_size,
                    train_num_samples=args.num_samples,
                    persistent=True,
                    num_workers=args.num_workers,
                )
                val_loader = get_loader_bone_tumor(
                    root_dir=args.data_root_path,
                    modality='MIX',
                    phase='val',
                    batch_size=1,
                    roi_size=roi_size,
                    train_num_samples=1,
                    persistent=True,
                    num_workers=args.num_workers,
                )
        else:
            train_loader = get_loader_bone_tumor(
                root_dir=args.data_root_path,
                modality=args.train_modality,
                phase='train',
                batch_size=args.batch_size,
                roi_size=roi_size,
                train_num_samples=args.num_samples,
                persistent=True,
                num_workers=args.num_workers,
            )
            val_loader = get_loader_bone_tumor(
                root_dir=args.data_root_path,
                modality=args.train_modality,
                phase='val',
                batch_size=1,
                roi_size=roi_size,
                train_num_samples=1,
                persistent=True,
                num_workers=args.num_workers,
            )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    use_cross_attention = getattr(args, 'use_cross_attention', False)
    if use_cross_attention and args.train_modality == 'MIX' and args.dataset == 'bone_tumor':
        print(f"[Dataset] Paired train samples: {len(train_loader_ct.dataset)}, Val samples: {len(val_loader.dataset)}")
    else:
        print(f"[Dataset] Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    if args.rank == 0:
        if args.with_text_embedding == 1:
            str_tmp = 'out/'+args.backbone+'/with_txt/CLIP_V3/' + args.log_name + '_' + args.train_modality + f'_lr{args.lr}' + f'_max_epoch{args.max_epoch}' + time.strftime('_%m_%d_%H_%M', time.localtime())
        else:
            str_tmp = 'out/'+args.backbone+'/no_txt/' + args.log_name + '_' + args.train_modality + f'_lr{args.lr}' + f'_max_epoch{args.max_epoch}' + time.strftime('_%m_%d_%H_%M', time.localtime())
        writer = SummaryWriter(log_dir=str_tmp)
        print('Writing Tensorboard logs to ', str_tmp)

    args.log_dir = str_tmp if args.rank == 0 else None

    best_dice = 0.0
    best_epoch = 0

    # Lists for plotting routing trends
    epochs_history = []
    e0_history = []
    e1_history = []

    while args.epoch < args.max_epoch:
        if scheduler is not None:
            scheduler.step()
        if args.train_modality == 'MIX':
            if args.dataset == 'data1':
                loss_dice, loss_bce, w0, w1, avg_route_ce = train_mix(args, train_loader_ct, train_loader_mr, model, optimizer, loss_function, loss_seg_CE)
            elif args.dataset == 'bone_tumor' and getattr(args, 'use_cross_attention', False):
                loss_dice, loss_bce, w0, w1, avg_route_ce = train_mix(args, train_loader_ct, train_loader_mr, model, optimizer, loss_function, loss_seg_CE)
            else:
                loss_dice, loss_bce, w0, w1 = train(args, train_loader, model, optimizer, loss_function, loss_seg_CE)
                avg_route_ce = 0.0 # 防止单模态时报错
        else:
            loss_dice, loss_bce, w0, w1 = train(args, train_loader, model, optimizer, loss_function, loss_seg_CE)
            avg_route_ce = 0.0 # 防止单模态时报错

        # Update plotting history
        epochs_history.append(args.epoch)
        e0_history.append(w0)
        e1_history.append(w1)

        if args.rank == 0 and len(epochs_history) > 1 and args.log_dir is not None:
            plot_path = os.path.join(args.log_dir, 'routing_trends.png')
            plot_routing_trends(epochs_history, e0_history, e1_history, plot_path)
            writer.add_scalar('route_weight_expert0', w0, args.epoch)
            writer.add_scalar('route_weight_expert1', w1, args.epoch)

        # Validation
        if args.dataset == 'bone_tumor':
            if args.rank == 0:
                vis_output_dir = os.path.join(args.log_dir, 'visualizations')
                os.makedirs(vis_output_dir, exist_ok=True)
            else:
                vis_output_dir = None

            # SWA update (epoch-level)
            if getattr(args, "swa_model", None) is not None and args.epoch >= args.swa_start:
                if (args.epoch - args.swa_start) % args.swa_update_every == 0:
                    args.swa_model.update_parameters(model)

            eval_state = None
            if getattr(args, "ema_model", None) is not None:
                eval_state = args.ema_model.state_dict()
            elif getattr(args, "swa_model", None) is not None and args.epoch >= args.swa_start:
                eval_state = args.swa_model.state_dict()

            if eval_state is not None:
                with _temporary_state_dict(model, eval_state):
                    val_metrics = enhanced_validation(
                        args, val_loader, model,
                        epoch=args.epoch,
                        output_dir=vis_output_dir,
                        loss=(loss_bce + loss_dice),
                        lr=optimizer.param_groups[0]['lr']
                    )
            else:
                val_metrics = enhanced_validation(
                    args, val_loader, model,
                    epoch=args.epoch,
                    output_dir=vis_output_dir,
                    loss=(loss_bce + loss_dice),
                    lr=optimizer.param_groups[0]['lr']
                )

            current_val_dice = val_metrics['foreground_dice_mean']
            print(f'\n[Epoch {args.epoch}] Current validation Dice (foreground): {current_val_dice:.8f}')
            print(f'[Epoch {args.epoch}] Precision: {val_metrics["precision_mean"]:.4f}, Recall: {val_metrics["recall_mean"]:.4f}, IoU: {val_metrics["iou_mean"]:.4f}')
            print(f'[Epoch {args.epoch}] Bucketed Dice - <2%: {val_metrics["bucket_lt2_dice"]:.4f}, 2-5%: {val_metrics["bucket_2to5_dice"]:.4f}, >5%: {val_metrics["bucket_gt5_dice"]:.4f}')
            print(f'[Epoch {args.epoch}] Best validation Dice so far: {best_dice:.8f} (at epoch {best_epoch})')

            if args.rank == 0:
                writer.add_scalar('val_dice_foreground', current_val_dice, args.epoch)
                writer.add_scalar('val_precision', val_metrics['precision_mean'], args.epoch)
                writer.add_scalar('val_recall', val_metrics['recall_mean'], args.epoch)
                writer.add_scalar('val_iou', val_metrics['iou_mean'], args.epoch)
                writer.add_scalar('val_iou_std', val_metrics['iou_std'], args.epoch)
                writer.add_scalar('val_voxel_iou', val_metrics['voxel_iou'], args.epoch)
                writer.add_scalar('val_dice_lt2', val_metrics['bucket_lt2_dice'], args.epoch)
                writer.add_scalar('val_dice_2to5', val_metrics['bucket_2to5_dice'], args.epoch)
                writer.add_scalar('val_dice_gt5', val_metrics['bucket_gt5_dice'], args.epoch)
                if np.isfinite(val_metrics.get('hd95_mean', float('nan'))):
                    writer.add_scalar('val_hd95', val_metrics['hd95_mean'], args.epoch)
                    writer.add_scalar('val_hd95_std', val_metrics['hd95_std'], args.epoch)
                if np.isfinite(val_metrics.get('assd_mean', float('nan'))):
                    writer.add_scalar('val_assd', val_metrics['assd_mean'], args.epoch)
                    writer.add_scalar('val_assd_std', val_metrics['assd_std'], args.epoch)
                writer.add_scalar('Route_CE', avg_route_ce, args.epoch)

                if current_val_dice > best_dice:
                    best_dice = current_val_dice
                    best_epoch = args.epoch
                    print(f'🎯 New best Dice: {best_dice:.8f} at epoch {best_epoch}')

                    best_net_state = model.state_dict()
                    if getattr(args, "ema_model", None) is not None:
                        best_net_state = args.ema_model.state_dict()
                    elif getattr(args, "swa_model", None) is not None and args.epoch >= args.swa_start:
                        best_net_state = args.swa_model.state_dict()

                    best_checkpoint = {
                        "net": best_net_state,
                        'optimizer': optimizer.state_dict(),
                        "epoch": args.epoch,
                        "best_dice": best_dice,
                        "val_metrics": val_metrics
                    }
                    if scheduler is not None:
                        best_checkpoint['scheduler'] = scheduler.state_dict()
                    if not os.path.isdir(str_tmp):
                        os.makedirs(str_tmp, exist_ok=True)
                    torch.save(best_checkpoint, str_tmp + '/best_model.pt')
                    print(f'✅ Best model saved!')

                    metrics_log_path = os.path.join(str_tmp, 'best_model_metrics.txt')
                    with open(metrics_log_path, 'w') as f:
                        f.write(f"[Best Model Validation Metrics - Epoch {args.epoch}]\n")
                        f.write(f"  Foreground Dice: {val_metrics['foreground_dice_mean']:.4f} ± {val_metrics['foreground_dice_std']:.4f}\n")
                        f.write(f"  Precision: {val_metrics['precision_mean']:.4f}\n")
                        f.write(f"  Recall: {val_metrics['recall_mean']:.4f}\n")
                        f.write(f"  IoU (non-empty GT): {val_metrics['iou_mean']:.4f} ± {val_metrics['iou_std']:.4f}\n")
                        f.write(f"  Voxel IoU @0.5: {val_metrics['voxel_iou']:.4f}\n")
                        if np.isfinite(val_metrics.get('hd95_mean', float('nan'))):
                            f.write(f"  HD95 (mm): {val_metrics['hd95_mean']:.4f} ± {val_metrics['hd95_std']:.4f}\n")
                            f.write(f"  ASSD (mm): {val_metrics['assd_mean']:.4f} ± {val_metrics['assd_std']:.4f}\n")
                        f.write(f"\n[Bucketed Dice by GT Positive Ratio]\n")
                        f.write(f"  <2%:   {val_metrics['bucket_lt2_dice']:.4f} (n={val_metrics['bucket_lt2_count']})\n")
                        f.write(f"  2-5%:  {val_metrics['bucket_2to5_dice']:.4f} (n={val_metrics['bucket_2to5_count']})\n")
                        f.write(f"  >5%:   {val_metrics['bucket_gt5_dice']:.4f} (n={val_metrics['bucket_gt5_count']})\n")

        if args.rank == 0:
            writer.add_scalar('train_dice_loss', loss_dice, args.epoch)
            writer.add_scalar('train_bce_loss', loss_bce, args.epoch)
            writer.add_scalar('total_loss', loss_bce + loss_dice, args.epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], args.epoch)

        should_save = (
            (args.rank == 0)
            and (
                (args.epoch % args.store_num == 0 and args.epoch != 0)
                or (args.max_epoch - args.epoch <= 5)
            )
        )
        if should_save:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                "epoch": args.epoch
            }
            if scheduler is not None:
                checkpoint['scheduler'] = scheduler.state_dict()
            if not os.path.isdir(str_tmp):
                os.makedirs(str_tmp, exist_ok=True)
            torch.save(checkpoint, str_tmp + '/epoch_' + str(args.epoch) + '.pt')
            print('save model success')

        args.epoch += 1

    if args.dataset == 'bone_tumor':
        print(f'\n{"="*60}')
        print(f'🏆 Training completed!')
        print(f'🎯 Best Dice: {best_dice:.4f} at epoch {best_epoch}')
        print(f'{"="*60}\n')

        if args.rank == 0:
            final_log_path = os.path.join(args.log_dir, 'training_summary.txt')
            with open(final_log_path, 'w') as f:
                f.write(f"[Training Summary]\n")
                f.write(f"  Loss Type: {args.loss_type}\n")
                if args.loss_type != 'dicece':
                    f.write(f"  Loss Parameters: alpha={args.loss_alpha}, beta={args.loss_beta}, gamma={args.loss_gamma}\n")
                f.write(f"  Max Epochs: {args.max_epoch}\n")
                f.write(f"  Learning Rate: {args.lr}\n")
                f.write(f"  Batch Size: {args.batch_size}\n")
                f.write(f"  Random Seed: {args.seed}\n")
                f.write(f"\n[Best Model]\n")
                f.write(f"  Best Dice: {best_dice:.4f}\n")
                f.write(f"  Best Epoch: {best_epoch}\n")
            print(f"[INFO] Training summary saved to: {final_log_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7], type=int, help="GPU device")
    parser.add_argument("--epoch", default=0)
    parser.add_argument('--log_name', default='unet', help='The path resume from checkpoint')
    parser.add_argument('--rank', default=0, type=int, help='use tensorboardX to log the training process')
    parser.add_argument('--backbone', default='unet', help='backbone [swinunetr or unet or dints or unetpp]')
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--finetune_from', default=None, help='Load model weights only (reset LR/optimizer/scheduler/epoch) and start a new training run')
    parser.add_argument('--pretrain', default=None, help='The path of pretrain model.')
    parser.add_argument('--trans_encoding', default='word_embedding', help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='./text_embedding/bone_tumor_class_embeddings.pth', help='The path of word embedding')
    parser.add_argument('--with_text_embedding', default=1, type=int, choices=[0, 1], help='whether use text embedding')
    parser.add_argument('--case_text_embedding', default=None, help='Path to patient-level caption embeddings generated by generate_embeddings.py')
    parser.add_argument('--use_case_text_embedding', action='store_true', help='Fuse static tumor embedding with per-case caption embedding')
    parser.add_argument('--case_text_alpha', type=float, default=0.3, help='Fusion weight for static_tumor + alpha * case_caption')
    parser.add_argument('--use_cross_attention', action='store_true', help='Use cross-attention between CT and MR')
    parser.add_argument('--cross_attn_heads', type=int, default=8, help='Number of attention heads for cross-attention')
    
    # ======== 新增：MoE Routing Entropy 权重控制 ========
    parser.add_argument('--entropy_weight', type=float, default=0.01, help='Weight for MoE routing entropy loss')
    # ====================================================

    # ======== EMA / SWA ========
    parser.add_argument('--use_ema', action='store_true', help='Enable EMA weights for evaluation/saving best')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay (higher = smoother)')
    parser.add_argument('--use_swa', action='store_true', help='Enable SWA weights for evaluation/saving best (epoch-level averaging)')
    parser.add_argument('--swa_start', type=int, default=150, help='Start epoch for SWA parameter averaging')
    parser.add_argument('--swa_update_every', type=int, default=1, help='Update SWA every N epochs after swa_start')
    # ===========================

    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=50, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--fixed_lr', action='store_true', help='Use fixed learning rate and disable LR scheduler')
    parser.add_argument('--boundary_loss_weight', type=float, default=0.0,
                        help='Weight for boundary-band Dice loss (0=disabled). Used in train_mix.')
    parser.add_argument('--boundary_kernel', type=int, default=3,
                        help='Odd kernel size for 3D morphological boundary band (default 3).')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')
    parser.add_argument('--dataset', default='data1', type=str, choices=['data1', 'bone_tumor'], help='Dataset to use: data1 (original) or bone_tumor')
    parser.add_argument('--data_root_path', default='./dataset/data1', help='data root path')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=2.0, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=2, type=int, help='sample number in each ct/mri scan')
    parser.add_argument('--num_class', default=2, type=int, help='the number of class for the segmentation (data1: 14, bone_tumor: 2)')
    parser.add_argument('--phase', default='train', help='train or val or test')
    parser.add_argument('--train_modality', default='MIX', type=str, choices=['CT', 'MR', 'MIX'], help='CT or MR or MIX')
    parser.add_argument('--loss_type', type=str, default='dicece', choices=['dicece', 'tversky', 'focal_tversky'], help='Loss function type')
    parser.add_argument('--loss_alpha', type=float, default=0.7, help='Tversky alpha parameter (FP weight)')
    parser.add_argument('--loss_beta', type=float, default=0.3, help='Tversky beta parameter (FN weight)')
    parser.add_argument('--loss_gamma', type=float, default=1.33, help='Focal gamma parameter')
    parser.add_argument('--pos_neg_ratio', type=float, default=None, help='Positive to negative patch ratio (e.g., 3.0 for 3:1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # ======== Pretrained encoder loading & freezing ========
    parser.add_argument('--pretrain_encoder_only', default=None,
                        help='Path to checkpoint; loads only encoder (swinViT.*) weights. '
                             'Handles SSL (encoder.* → swinViT.*) and BTCV/MONAI (swinViT.*) formats.')
    parser.add_argument('--freeze_level', default='none',
                        choices=['all', 'stage4', 'stage34', 'none'],
                        help='Encoder freeze strategy after loading pretrained weights. '
                             'all=freeze stages 1-4, stage4=freeze 1-3 thaw 4, '
                             'stage34=freeze 1-2 thaw 3-4, none=full fine-tune.')
    # =======================================================

    # ======== Fold-based data split ========
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold index (0-4) for 5-fold cross-validation split (splits/fold5_splits.json).')
    # =======================================

    args = parser.parse_args()

    process(args=args)

if __name__ == "__main__":
    main()