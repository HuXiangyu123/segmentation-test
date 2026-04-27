#!/usr/bin/env python
"""
Configuration loader for experiments.

Loads experiment configurations from YAML file and provides
easy access to experiment parameters.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ExperimentConfig:
    """Experiment configuration manager."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML config file. If None, uses default.
        """
        if config_path is None:
            # Default config path - go up to bone_tumor directory
            config_path = Path(__file__).parent.parent.parent / 'configs' / 'experiments.yaml'

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def get_common_params(self) -> Dict[str, Any]:
        """Get common parameters shared across all experiments."""
        return self.config.get('common', {})

    def get_experiment_config(self, exp_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific experiment.

        Args:
            exp_name: Experiment name (e.g., 'E0_baseline')

        Returns:
            Dictionary with experiment configuration
        """
        experiments = self.config.get('experiments', {})

        if exp_name not in experiments:
            available = list(experiments.keys())
            raise ValueError(f"Experiment '{exp_name}' not found. Available: {available}")

        # Merge common params with experiment-specific params
        exp_config = experiments[exp_name].copy()

        # Add common params
        exp_config['common'] = self.get_common_params()

        return exp_config

    def get_all_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Get all experiment configurations."""
        return self.config.get('experiments', {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get('evaluation', {})

    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration."""
        return self.config.get('analysis', {})

    def build_train_args(self, exp_name: str) -> Dict[str, Any]:
        """
        Build training arguments for an experiment.

        Args:
            exp_name: Experiment name

        Returns:
            Dictionary with all training arguments
        """
        exp_config = self.get_experiment_config(exp_name)
        common = exp_config['common']

        # Build args dictionary
        args = {
            # Common params
            'dataset': common['dataset'],
            'data_root_path': common['data_root_path'],
            'backbone': common['backbone'],
            'with_text_embedding': common['with_text_embedding'],
            'num_class': common['num_class'],
            'train_modality': common['train_modality'],
            'max_epoch': common['max_epoch'],
            'batch_size': common['batch_size'],
            'num_workers': common['num_workers'],
            'lr': common['lr'],
            'warmup_epoch': common['warmup_epoch'],
            'seed': common['seed'],
            'device': common['device'],
            'roi_x': common['roi_x'],
            'roi_y': common['roi_y'],
            'roi_z': common['roi_z'],

            # Experiment-specific params
            'log_name': exp_config['log_name'],
            'loss_type': exp_config['loss']['type'],
        }

        # Add loss parameters
        loss_params = exp_config['loss'].get('params', {})
        for key, value in loss_params.items():
            args[f'loss_{key}'] = value

        # Add sampling parameters
        sampling = exp_config.get('sampling', {})
        if sampling.get('train_num_samples'):
            args['num_samples'] = sampling['train_num_samples']
        if sampling.get('pos_neg_ratio'):
            args['pos_neg_ratio'] = sampling['pos_neg_ratio']

        # Add cross-attention parameters
        cross_attention = exp_config.get('cross_attention', {})
        if cross_attention.get('use_cross_attention', False):
            args['use_cross_attention'] = True
        if cross_attention.get('cross_attn_heads'):
            args['cross_attn_heads'] = cross_attention['cross_attn_heads']

        return args

    def build_command(self, exp_name: str) -> list:
        """
        Build command line for running an experiment.

        Args:
            exp_name: Experiment name

        Returns:
            List of command arguments
        """
        args = self.build_train_args(exp_name)

        cmd = ['python', 'train.py']

        # Add all arguments
        for key, value in args.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        cmd.append(f'--{key}')
                else:
                    cmd.extend([f'--{key}', str(value)])

        return cmd

    def print_experiment_summary(self, exp_name: str):
        """Print a summary of experiment configuration."""
        exp_config = self.get_experiment_config(exp_name)

        print(f"\n{'='*80}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*80}")
        print(f"Description: {exp_config['description']}")
        print(f"\nLoss Configuration:")
        print(f"  Type: {exp_config['loss']['type']}")
        if exp_config['loss'].get('params'):
            print(f"  Parameters:")
            for key, value in exp_config['loss']['params'].items():
                print(f"    {key}: {value}")

        print(f"\nSampling Configuration:")
        sampling = exp_config.get('sampling', {})
        print(f"  train_num_samples: {sampling.get('train_num_samples', 1)}")
        print(f"  pos_neg_ratio: {sampling.get('pos_neg_ratio', 'default')}")

        print(f"\nExpected Outcomes:")
        for target in exp_config.get('targets', []):
            print(f"  - {target}")

        print(f"{'='*80}\n")


def main():
    """Test configuration loader."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all',
                        help='Experiment to show (or "all")')
    parser.add_argument('--show_command', action='store_true',
                        help='Show command line')
    args = parser.parse_args()

    # Load config
    config = ExperimentConfig()

    if args.exp == 'all':
        # Show all experiments
        experiments = config.get_all_experiments()
        for exp_name in experiments.keys():
            config.print_experiment_summary(exp_name)
            if args.show_command:
                cmd = config.build_command(exp_name)
                print(f"Command: {' '.join(cmd)}\n")
    else:
        # Show specific experiment
        config.print_experiment_summary(args.exp)
        if args.show_command:
            cmd = config.build_command(args.exp)
            print(f"Command: {' '.join(cmd)}\n")


if __name__ == '__main__':
    main()
