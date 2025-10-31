import numpy as np
from typing import List
from core.entities import Task

class TaskGenerator:
    """Generate tasks with various characteristics"""
    
    def __init__(self, config: dict):
        self.config = config
        self.task_counter = 0
        
    def generate_task(self, vehicle_id: int, current_time: float) -> Task:
        """Generate a single task"""
        self.task_counter += 1
        
        task = Task(
            id=self.task_counter,
            data_size=np.random.uniform(
                *self.config['tasks']['data_size_range']
            ),
            computation_demand=np.random.uniform(
                *self.config['tasks']['computation_demand_range']
            ),
            required_service=self._select_service_zipf(),
            generation_time=current_time,
            deadline=current_time + np.random.exponential(5),  # Exponential deadline
            source_vehicle=vehicle_id
        )
        
        return task
    
    def _select_service_zipf(self) -> int:
        """Select service based on Zipf distribution"""
        num_services = self.config['services']['total_services']
        alpha = self.config['services']['zipf_parameter']
        
        # Generate Zipf distribution
        probabilities = np.array([1/k**alpha for k in range(1, num_services+1)])
        probabilities /= probabilities.sum()
        
        return np.random.choice(num_services, p=probabilities)
    
    def generate_batch(self, vehicle_ids: List[int], 
                      current_time: float, 
                      generation_prob: float = 0.7) -> List[Task]:
        """Generate batch of tasks for multiple vehicles"""
        tasks = []
        
        for vid in vehicle_ids:
            if np.random.random() < generation_prob:
                task = self.generate_task(vid, current_time)
                tasks.append(task)
                
        return tasks