"""
Population-Based Training (PBT) Scheduler
Auto-tunes hyperparameters during training
"""
import numpy as np
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
import copy


@dataclass
class Population:
    """Individual in the population"""
    id: int
    hyperparams: Dict
    performance: float
    age: int
    model_state: Optional[Dict] = None


class PBTScheduler:
    """Population-Based Training scheduler for hyperparameter optimization"""
    
    def __init__(
        self,
        population_size: int = 8,
        exploit_interval: int = 5,
        truncation_threshold: float = 0.2,
        perturbation_factors: tuple = (0.8, 1.2)
    ):
        self.population_size = population_size
        self.exploit_interval = exploit_interval
        self.truncation_threshold = truncation_threshold
        self.perturbation_factors = perturbation_factors
        
        self.population: List[Population] = []
        self.generation = 0
        self.history = []
        
    def initialize_population(self, base_hyperparams: Dict) -> List[Population]:
        """Initialize population with random hyperparameter variations"""
        self.population = []
        
        # Hyperparameter search space
        search_space = {
            'learning_rate': (1e-5, 1e-3),
            'gamma': (0.95, 0.999),
            'gae_lambda': (0.9, 0.99),
            'clip_param': (0.1, 0.3),
            'entropy_coef': (0.0, 0.1),
            'value_loss_coef': (0.25, 1.0),
            'max_grad_norm': (0.3, 1.0),
            'batch_size': [128, 256, 512, 1024],
            'n_steps': [256, 512, 1024, 2048]
        }
        
        for i in range(self.population_size):
            hyperparams = copy.deepcopy(base_hyperparams)
            
            # Sample from search space
            for key, value_range in search_space.items():
                if isinstance(value_range, list):
                    hyperparams[key] = random.choice(value_range)
                else:
                    low, high = value_range
                    # Use log-uniform sampling for positive ranges, linear for others
                    if low > 0 and high > low:
                        if key in ['batch_size', 'n_steps']:
                            hyperparams[key] = int(np.exp(np.random.uniform(np.log(low), np.log(high))))
                        else:
                            hyperparams[key] = np.exp(np.random.uniform(np.log(low), np.log(high)))
                    else:
                        # Fallback to linear sampling
                        if key in ['batch_size', 'n_steps']:
                            hyperparams[key] = int(np.random.uniform(low, high))
                        else:
                            hyperparams[key] = np.random.uniform(low, high)
            
            self.population.append(Population(
                id=i,
                hyperparams=hyperparams,
                performance=0.0,
                age=0
            ))
        
        return self.population
    
    def step(self, performances: Dict[int, float]) -> List[Population]:
        """
        Execute one PBT step: exploit and explore
        
        Args:
            performances: Dict mapping population_id to performance metric (higher is better)
        """
        # Update performances
        for pop in self.population:
            if pop.id in performances:
                pop.performance = performances[pop.id]
                pop.age += 1
        
        # Sort by performance
        self.population.sort(key=lambda x: x.performance, reverse=True)
        
        # Exploit and explore
        if self.generation % self.exploit_interval == 0 and self.generation > 0:
            self._exploit_and_explore()
        
        self.generation += 1
        
        # Log history
        self.history.append({
            'generation': self.generation,
            'best_performance': self.population[0].performance,
            'mean_performance': np.mean([p.performance for p in self.population]),
            'worst_performance': self.population[-1].performance,
            'best_hyperparams': self.population[0].hyperparams
        })
        
        return self.population
    
    def _exploit_and_explore(self):
        """Exploit: copy top performers, Explore: perturb hyperparameters"""
        n_truncate = int(self.population_size * self.truncation_threshold)
        
        # Bottom performers copy from top performers
        for i in range(self.population_size - n_truncate, self.population_size):
            # Copy from a random top performer
            top_idx = random.randint(0, n_truncate - 1)
            
            print(f"🔄 PBT: Population {i} (perf={self.population[i].performance:.3f}) "
                  f"copying from {top_idx} (perf={self.population[top_idx].performance:.3f})")
            
            # Copy hyperparameters
            self.population[i].hyperparams = copy.deepcopy(
                self.population[top_idx].hyperparams
            )
            
            # Copy model state if available
            if self.population[top_idx].model_state:
                self.population[i].model_state = copy.deepcopy(
                    self.population[top_idx].model_state
                )
            
            # Perturb hyperparameters (explore)
            self._perturb_hyperparams(self.population[i])
            
            # Reset age and performance
            self.population[i].age = 0
            self.population[i].performance = 0.0
    
    def _perturb_hyperparams(self, individual: Population):
        """Randomly perturb hyperparameters"""
        perturbable = [
            'learning_rate', 'gamma', 'gae_lambda', 'clip_param',
            'entropy_coef', 'value_loss_coef', 'max_grad_norm'
        ]
        
        for key in perturbable:
            if key in individual.hyperparams:
                if random.random() < 0.5:  # 50% chance to perturb each param
                    factor = random.choice(self.perturbation_factors)
                    old_value = individual.hyperparams[key]
                    individual.hyperparams[key] *= factor
                    
                    # Clip to reasonable ranges
                    if key == 'learning_rate':
                        individual.hyperparams[key] = np.clip(
                            individual.hyperparams[key], 1e-6, 1e-2
                        )
                    elif key == 'gamma':
                        individual.hyperparams[key] = np.clip(
                            individual.hyperparams[key], 0.9, 0.999
                        )
                    elif key == 'clip_param':
                        individual.hyperparams[key] = np.clip(
                            individual.hyperparams[key], 0.05, 0.5
                        )
                    
                    print(f"  ↪️ Perturbed {key}: {old_value:.6f} → "
                          f"{individual.hyperparams[key]:.6f}")
    
    def get_best(self) -> Population:
        """Get best performing individual"""
        return self.population[0]
    
    def get_stats(self) -> Dict:
        """Get population statistics"""
        performances = [p.performance for p in self.population]
        ages = [p.age for p in self.population]
        
        return {
            'generation': self.generation,
            'best_performance': max(performances),
            'mean_performance': np.mean(performances),
            'std_performance': np.std(performances),
            'mean_age': np.mean(ages),
            'population_size': len(self.population)
        }
    
    def save_state(self, filepath: str):
        """Save PBT state"""
        import pickle
        state = {
            'population': self.population,
            'generation': self.generation,
            'history': self.history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str):
        """Load PBT state"""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.population = state['population']
        self.generation = state['generation']
        self.history = state['history']


class AdaptivePBTScheduler(PBTScheduler):
    """
    Adaptive PBT that adjusts exploitation frequency based on performance variance
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_history = []
        self.adaptive_exploit_interval = self.exploit_interval
    
    def step(self, performances: Dict[int, float]) -> List[Population]:
        """Step with adaptive exploitation"""
        # Update performances
        for pop in self.population:
            if pop.id in performances:
                pop.performance = performances[pop.id]
                pop.age += 1
        
        # Track performance variance
        current_performances = [p.performance for p in self.population]
        self.performance_history.append(np.std(current_performances))
        
        # Adapt exploitation interval based on variance
        if len(self.performance_history) > 10:
            recent_variance = np.mean(self.performance_history[-10:])
            
            # High variance = explore more (longer intervals)
            # Low variance = exploit more (shorter intervals)
            if recent_variance > 0.1:
                self.adaptive_exploit_interval = max(10, self.exploit_interval + 5)
            elif recent_variance < 0.05:
                self.adaptive_exploit_interval = max(3, self.exploit_interval - 2)
        
        # Sort by performance
        self.population.sort(key=lambda x: x.performance, reverse=True)
        
        # Exploit and explore with adaptive interval
        if self.generation % self.adaptive_exploit_interval == 0 and self.generation > 0:
            print(f"🔧 Adaptive PBT: Using exploit interval = {self.adaptive_exploit_interval}")
            self._exploit_and_explore()
        
        self.generation += 1
        
        # Log history
        self.history.append({
            'generation': self.generation,
            'best_performance': self.population[0].performance,
            'mean_performance': np.mean(current_performances),
            'exploit_interval': self.adaptive_exploit_interval,
            'performance_variance': np.std(current_performances)
        })
        
        return self.population
