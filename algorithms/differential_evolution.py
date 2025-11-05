import numpy as np
from typing import Callable, Tuple

class DifferentialEvolution:
    """Differential Evolution operations for Cuckoo Search"""
    
    def __init__(self, scaling_factor: float = 0.5, 
                 crossover_prob: float = 0.9):
        self.scaling_factor = scaling_factor
        self.crossover_prob = crossover_prob
        
    def mutation(self, population: np.ndarray, 
                 target_idx: int) -> np.ndarray:
        """DE/rand/1 mutation strategy"""
        pop_size, dim = population.shape
        
        # Select three random distinct indices
        indices = list(range(pop_size))
        indices.remove(target_idx)
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        
        # Mutation vector
        mutant = population[r1] + self.scaling_factor * \
                 (population[r2] - population[r3])
        
        # Ensure bounds [0, 1]
        return np.clip(mutant, 0, 1)
    
    def crossover(self, target: np.ndarray, 
                  mutant: np.ndarray) -> np.ndarray:
        """Binomial crossover"""
        dim = len(target)
        trial = np.copy(target)
        
        # Ensure at least one parameter from mutant
        j_rand = np.random.randint(dim)
        
        for j in range(dim):
            if np.random.random() < self.crossover_prob or j == j_rand:
                trial[j] = mutant[j]
                
        return trial
    
    def selection(self, target: np.ndarray, trial: np.ndarray,
                  fitness_func: Callable) -> Tuple[np.ndarray, float]:
        """Greedy selection"""
        target_fitness = fitness_func(target)
        trial_fitness = fitness_func(trial)
        
        if trial_fitness < target_fitness:
            return trial, trial_fitness
        else:
            return target, target_fitness
    
    def evolve_population(self, population: np.ndarray,
                          fitness_func: Callable) -> np.ndarray:
        """Evolve entire population for one generation"""
        pop_size = len(population)
        new_population = np.copy(population)
        
        for i in range(pop_size):
            # Mutation
            mutant = self.mutation(population, i)
            
            # Crossover
            trial = self.crossover(population[i], mutant)
            
            # Selection
            new_population[i], _ = self.selection(
                population[i], trial, fitness_func
            )
            
        return new_population