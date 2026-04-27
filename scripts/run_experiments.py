#!/usr/bin/env python
"""
Experiment runner for loss and sampling optimization.

Experiments:
- E0: Baseline (current setup)
- E1: TverskyLoss(alpha=0.7, beta=0.3)
- E2: FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)
- E3: Enhanced sampling (pos:neg=3:1, increased num_samples)
- E6: Boundary-band Dice loss (emphasize tumor edges) + cross-attention
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import json


# Experiment configurations
EXPERIMENTS = {
    'E0_baseline': {
        'description': 'Baseline (DiceCE loss, default sampling)',
        'loss_type': 'dicece',
        'loss_params': {},
        'sampling_params': {
            'pos_neg_ratio': None,  # default
            'num_samples': 1,  # default
        }
    },
    'E1_tversky': {
        'description': 'TverskyLoss(alpha=0.7, beta=0.3)',
        'loss_type': 'tversky',
        'loss_params': {
            'alpha': 0.7,
            'beta': 0.3,
        },
        'sampling_params': {
            'pos_neg_ratio': None,
            'num_samples': 1,
        }
    },
    'E2_focal_tversky': {
        'description': 'FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)',
        'loss_type': 'focal_tversky',
        'loss_params': {
            'alpha': 0.7,
            'beta': 0.3,
            'gamma': 1.33,
        },
        'sampling_params': {
            'pos_neg_ratio': None,
            'num_samples': 1,
        }
    },
    'E3_enhanced_sampling': {
        'description': 'Enhanced sampling (pos:neg=3:1, num_samples=4)',
        'loss_type': 'dicece',
        'loss_params': {},
        'sampling_params': {
            'pos_neg_ratio': 3.0,  # 3:1 ratio
            'num_samples': 4,  # 4x more samples per epoch
        }
    },
    'E4_cross_attention': {
        'description': 'Cross-attention between CT and MR (heads=8)',
        # 'loss_type': 'focal_tversky',
        # 'loss_params': {
        #     'alpha': 0.8,
        #     'beta': 0.2,
        #     'gamma': 1.3,
        # },
        'loss_type': 'dicece',
        'loss_params': {},
        'sampling_params': {
            'pos_neg_ratio': 3,
            'num_samples': 1,
        },
        'cross_attention_params': {
            'use_cross_attention': True,
            'cross_attn_heads': 8,
        },
        'pretrain': '/root/autodl-tmp/segmentation-test-main/MulModSeg_2024/pretrained_weights/model_swinvit.pt',
        'finetune_from': '/root/autodl-tmp/segmentation-test-main/MulModSeg_2024/out/swinunetr/with_txt/CLIP_V3/E4_cross_attention_MIX_lr0.0001_max_epoch200_03_27_13_15/best_model.pt',
    },
    'E5_finetune_ema_swa': {
        'description': 'Finetune from Dice~0.82 best_model with fixed LR=2e-5 + EMA',
        'loss_type': 'dicece',
        'loss_params': {},
        'sampling_params': {
            'pos_neg_ratio': 3,
            'num_samples': 1,
        },
        'cross_attention_params': {
            'use_cross_attention': True,
            'cross_attn_heads': 8,
        },
        'pretrain': '/root/autodl-tmp/segmentation-test-main/MulModSeg_2024/pretrained_weights/model_swinvit.pt',
        'finetune_from': '/root/autodl-tmp/segmentation-test-main/best_model_082.pt',
        'lr': 2e-5,
        'fixed_lr': True,
        'warmup_epoch': 1,
        'max_epoch': 50,
        'use_ema': True,
        'ema_decay': 0.999,
        'use_swa': False,
    },
    'E6_boundary': {
        'description': 'Cross-attention + boundary-band Dice on GT tumor edge (finetune + fixed LR + EMA)',
        'loss_type': 'dicece',
        'loss_params': {},
        'sampling_params': {
            'pos_neg_ratio': 3,
            'num_samples': 1,
        },
        'cross_attention_params': {
            'use_cross_attention': True,
            'cross_attn_heads': 8,
        },
        'pretrain': '/root/autodl-tmp/segmentation-test-main/MulModSeg_2024/pretrained_weights/model_swinvit.pt',
        'finetune_from': '/root/autodl-tmp/segmentation-test-main/best_model_082.pt',
        'lr': 2e-5,
        'fixed_lr': True,
        'warmup_epoch': 1,
        'max_epoch': 50,
        'use_ema': True,
        'ema_decay': 0.999,
        'use_swa': False,
        'boundary_loss_weight': 0.5,
        'boundary_kernel': 3,
    },
}


# Common training parameters
COMMON_PARAMS = {
    'data_root_path': '/root/autodl-tmp/segmentation-test-main/MulModSeg_2024/dataset',
    'device': 0,
    'backbone': 'swinunetr',
    'with_text_embedding': 1,
    'train_modality': 'MIX',
    'max_epoch': 200,
    'batch_size': 1,
    'num_workers': 8,
    'num_class': 2,
    'lr': 1e-4,
    'warmup_epoch': 2,
    'seed': 42,  # Fixed seed for reproducibility
}


def build_command(exp_name, exp_config):
    """Build training command for an experiment."""
    lr = exp_config.get('lr', COMMON_PARAMS['lr'])
    warmup_epoch = exp_config.get('warmup_epoch', COMMON_PARAMS['warmup_epoch'])
    max_epoch = exp_config.get('max_epoch', COMMON_PARAMS['max_epoch'])
    cmd = [
        'python', 'train.py',
        '--dataset', 'bone_tumor',
        '--data_root_path', COMMON_PARAMS['data_root_path'],
        '--device', str(COMMON_PARAMS['device']),
        '--backbone', COMMON_PARAMS['backbone'],
        '--with_text_embedding', str(COMMON_PARAMS['with_text_embedding']),
        '--train_modality', COMMON_PARAMS['train_modality'],
        '--max_epoch', str(max_epoch),
        '--batch_size', str(COMMON_PARAMS['batch_size']),
        '--num_workers', str(COMMON_PARAMS['num_workers']),
        '--num_class', str(COMMON_PARAMS['num_class']),
        '--lr', str(lr),
        '--warmup_epoch', str(warmup_epoch),
        '--seed', str(COMMON_PARAMS['seed']),
        '--log_name', exp_name,
        '--loss_type', exp_config['loss_type'],
    ]

    # Add loss parameters
    for key, value in exp_config['loss_params'].items():
        cmd.extend([f'--loss_{key}', str(value)])

    # Add sampling parameters
    if exp_config['sampling_params']['pos_neg_ratio'] is not None:
        cmd.extend(['--pos_neg_ratio', str(exp_config['sampling_params']['pos_neg_ratio'])])
    if exp_config['sampling_params']['num_samples'] != 1:
        cmd.extend(['--num_samples', str(exp_config['sampling_params']['num_samples'])])

    # Add cross-attention parameters
    if 'cross_attention_params' in exp_config:
        if exp_config['cross_attention_params'].get('use_cross_attention', False):
            cmd.append('--use_cross_attention')
        if 'cross_attn_heads' in exp_config['cross_attention_params']:
            cmd.extend(['--cross_attn_heads', str(exp_config['cross_attention_params']['cross_attn_heads'])])

    # Add pretrain weights
    if 'pretrain' in exp_config:
        cmd.extend(['--pretrain', exp_config['pretrain']])

    # Add finetune weights (load model only; reset LR/optimizer/scheduler)
    if 'finetune_from' in exp_config and exp_config['finetune_from']:
        cmd.extend(['--finetune_from', exp_config['finetune_from']])

    # EMA / SWA flags
    if exp_config.get('use_ema', False):
        cmd.append('--use_ema')
        cmd.extend(['--ema_decay', str(exp_config.get('ema_decay', 0.999))])
    if exp_config.get('fixed_lr', False):
        cmd.append('--fixed_lr')
    if exp_config.get('use_swa', False):
        cmd.append('--use_swa')
        cmd.extend(['--swa_start', str(exp_config.get('swa_start', 150))])
        cmd.extend(['--swa_update_every', str(exp_config.get('swa_update_every', 1))])

    if exp_config.get('boundary_loss_weight', 0) and float(exp_config['boundary_loss_weight']) > 0:
        cmd.extend(['--boundary_loss_weight', str(exp_config['boundary_loss_weight'])])
        if 'boundary_kernel' in exp_config:
            cmd.extend(['--boundary_kernel', str(exp_config['boundary_kernel'])])

    return cmd


def run_experiment(exp_name, exp_config, dry_run=False):
    """Run a single experiment."""
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"Description: {exp_config['description']}")
    print(f"{'='*80}\n")

    cmd = build_command(exp_name, exp_config)
    print(f"Command: {' '.join(cmd)}\n")

    if dry_run:
        print("[DRY RUN] Skipping actual execution\n")
        return

    # Run the experiment
    result = subprocess.run(cmd, cwd='/root/autodl-tmp/segmentation-test-main/MulModSeg_2024')

    if result.returncode != 0:
        print(f"\n[ERROR] Experiment {exp_name} failed with return code {result.returncode}")
        return False

    print(f"\n[SUCCESS] Experiment {exp_name} completed successfully")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all',
                        help='Experiment to run (E0..E6, or all)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing')
    args = parser.parse_args()

    # Determine which experiments to run
    if args.exp == 'all':
        exp_list = list(EXPERIMENTS.keys())
    else:
        exp_key = f'{args.exp}_' + ('baseline' if args.exp == 'E0' else
                                     'tversky' if args.exp == 'E1' else
                                     'focal_tversky' if args.exp == 'E2' else
                                     'enhanced_sampling' if args.exp == 'E3' else
                                     'cross_attention' if args.exp == 'E4' else
                                     'finetune_ema_swa' if args.exp == 'E5' else
                                     'boundary')
        if exp_key not in EXPERIMENTS:
            print(f"[ERROR] Unknown experiment: {args.exp}")
            print(f"Available: {list(EXPERIMENTS.keys())}")
            sys.exit(1)
        exp_list = [exp_key]

    print(f"\n{'='*80}")
    print(f"Running {len(exp_list)} experiment(s)")
    print(f"{'='*80}\n")

    # Run experiments
    results = {}
    for exp_name in exp_list:
        success = run_experiment(exp_name, EXPERIMENTS[exp_name], dry_run=args.dry_run)
        results[exp_name] = success

    # Summary
    print(f"\n{'='*80}")
    print("Experiment Summary")
    print(f"{'='*80}\n")
    for exp_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{exp_name}: {status}")

    print(f"\n{'='*80}")
    print("All experiments completed!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
