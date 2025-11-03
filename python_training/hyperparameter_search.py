#!/usr/bin/env python3
"""
Hyperparameter Optimization with Optuna
Systematic search for optimal training configurations
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import logging
from typing import Dict, Callable
import numpy as np

logger = logging.getLogger(__name__)


class HyperparameterSearch:
    """
    Optuna-based hyperparameter optimization for RL trading policies
    """
    
    def __init__(
        self,
        study_name: str = "rl_trading_hpo",
        storage: str = None,  # Can be database URL for distributed search
        n_trials: int = 50,
        n_jobs: int = 1,  # Set to > 1 for parallel search
        seed: int = 42
    ):
        self.study_name = study_name
        self.storage = storage
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.seed = seed
        
        self.sampler = TPESampler(seed=seed)
        self.pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    
    def suggest_bc_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """
        Suggest hyperparameters for Behavior Cloning phase
        """
        return {
            'bc_lr': trial.suggest_float('bc_lr', 1e-5, 5e-4, log=True),
            'bc_batch_size': trial.suggest_categorical('bc_batch_size', [256, 512, 1024]),
            'bc_weight_decay': trial.suggest_float('bc_weight_decay', 1e-6, 1e-4, log=True),
            'bc_hidden_dim': trial.suggest_categorical('bc_hidden_dim', [256, 512, 1024]),
            'bc_n_layers': trial.suggest_int('bc_n_layers', 2, 5),
            'bc_dropout': trial.suggest_float('bc_dropout', 0.0, 0.3),
        }
    
    def suggest_ppo_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """
        Suggest hyperparameters for PPO phase
        """
        return {
            'ppo_learning_rate': trial.suggest_float('ppo_learning_rate', 1e-5, 5e-4, log=True),
            'ppo_n_steps': trial.suggest_categorical('ppo_n_steps', [2048, 4096, 8192]),
            'ppo_batch_size': trial.suggest_categorical('ppo_batch_size', [2048, 4096, 8192]),
            'ppo_gamma': trial.suggest_float('ppo_gamma', 0.98, 0.999),
            'ppo_gae_lambda': trial.suggest_float('ppo_gae_lambda', 0.95, 0.99),
            'ppo_clip_range': trial.suggest_float('ppo_clip_range', 0.1, 0.3),
            'ppo_ent_coef': trial.suggest_float('ppo_ent_coef', 0.001, 0.01, log=True),
            'ppo_vf_coef': trial.suggest_float('ppo_vf_coef', 0.5, 1.0),
            'ppo_max_grad_norm': trial.suggest_float('ppo_max_grad_norm', 0.3, 1.0),
        }
    
    def suggest_reward_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """
        Suggest hyperparameters for reward shaping
        """
        return {
            'lambda_risk': trial.suggest_float('lambda_risk', 0.1, 0.5),
            'sharpe_bonus_scale': trial.suggest_float('sharpe_bonus_scale', 0.5, 2.0),
            'drawdown_penalty_scale': trial.suggest_float('drawdown_penalty_scale', 0.5, 2.0),
        }
    
    def suggest_architecture_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """
        Suggest hyperparameters for network architecture
        """
        return {
            'frame_stack_size': trial.suggest_categorical('frame_stack_size', [30, 60, 90, 120]),
            'hidden_dims': trial.suggest_categorical('hidden_dims', [
                [256, 256],
                [512, 256],
                [512, 512],
                [1024, 512],
                [1024, 512, 256]
            ]),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'elu']),
        }
    
    def create_objective(
        self,
        train_fn: Callable,
        metric_name: str = 'val_sharpe',
        mode: str = 'max'
    ) -> Callable:
        """
        Create objective function for Optuna
        
        Args:
            train_fn: Function that takes hyperparameters and returns metrics dict
            metric_name: Name of metric to optimize
            mode: 'max' or 'min'
        """
        def objective(trial: optuna.Trial) -> float:
            # Suggest all hyperparameters
            hyperparameters = {}
            hyperparameters.update(self.suggest_bc_hyperparameters(trial))
            hyperparameters.update(self.suggest_ppo_hyperparameters(trial))
            hyperparameters.update(self.suggest_reward_hyperparameters(trial))
            hyperparameters.update(self.suggest_architecture_hyperparameters(trial))
            
            try:
                # Run training with suggested hyperparameters
                logger.info(f"Trial {trial.number}: Testing hyperparameters")
                logger.info(f"Hyperparameters: {hyperparameters}")
                
                metrics = train_fn(hyperparameters, trial)
                
                # Extract target metric
                target_value = metrics.get(metric_name)
                
                if target_value is None:
                    raise ValueError(f"Metric '{metric_name}' not found in results")
                
                logger.info(f"Trial {trial.number}: {metric_name} = {target_value:.4f}")
                
                return target_value if mode == 'max' else -target_value
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                raise optuna.TrialPruned()
        
        return objective
    
    def optimize(
        self,
        train_fn: Callable,
        metric_name: str = 'val_sharpe',
        mode: str = 'max'
    ) -> optuna.Study:
        """
        Run hyperparameter optimization
        
        Args:
            train_fn: Training function that accepts hyperparameters
            metric_name: Metric to optimize
            mode: 'max' or 'min'
        
        Returns:
            Completed Optuna study
        """
        direction = 'maximize' if mode == 'max' else 'minimize'
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self.sampler,
            pruner=self.pruner,
            direction=direction,
            load_if_exists=True
        )
        
        objective = self.create_objective(train_fn, metric_name, mode)
        
        logger.info(f"Starting hyperparameter search: {self.n_trials} trials, {self.n_jobs} workers")
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best {metric_name}: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters: {study.best_params}")
        
        return study
    
    def get_best_hyperparameters(self, study: optuna.Study) -> Dict:
        """Extract best hyperparameters from completed study"""
        return study.best_params
    
    def get_top_k_trials(self, study: optuna.Study, k: int = 5) -> list:
        """Get top K best trials for ensemble"""
        sorted_trials = sorted(
            study.trials,
            key=lambda t: t.value if t.value is not None else float('-inf'),
            reverse=True
        )
        return sorted_trials[:k]


# Preset hyperparameter configurations for quick testing
PRESET_CONFIGS = {
    'fast': {
        'bc_epochs': 1000,
        'bc_lr': 1e-4,
        'bc_batch_size': 512,
        'ppo_total_timesteps': 100_000,
        'ppo_n_steps': 2048,
        'ppo_batch_size': 2048,
        'frame_stack_size': 30,
    },
    'balanced': {
        'bc_epochs': 3000,
        'bc_lr': 2e-4,
        'bc_batch_size': 512,
        'ppo_total_timesteps': 500_000,
        'ppo_n_steps': 4096,
        'ppo_batch_size': 4096,
        'frame_stack_size': 60,
    },
    'aggressive': {
        'bc_epochs': 5000,
        'bc_lr': 3e-4,
        'bc_batch_size': 1024,
        'ppo_total_timesteps': 2_000_000,
        'ppo_n_steps': 8192,
        'ppo_batch_size': 8192,
        'frame_stack_size': 120,
    }
}


def get_preset_config(preset_name: str = 'balanced') -> Dict:
    """Get a preset configuration for quick training"""
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Choose from {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[preset_name]
