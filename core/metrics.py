import numpy as np
from typing import List, Dict
from core.entities import AccessPoint, Task

class MetricsCalculator:
    """Calculate various performance metrics for ECORA"""
    
    @staticmethod
    def calculate_average_delay(delays: List[float]) -> float:
        """Calculate average task processing delay"""
        return np.mean(delays) if delays else 0.0
    
    @staticmethod
    def calculate_load_balance(ap_loads: Dict[int, int]) -> float:
        """Calculate load balance metric (lower is better)"""
        if not ap_loads:
            return 0.0
        loads = list(ap_loads.values())
        return np.std(loads)  # Standard deviation as balance metric
    
    @staticmethod
    def calculate_throughput(completed_tasks: int, time_period: float) -> float:
        """Calculate system throughput"""
        return completed_tasks / time_period if time_period > 0 else 0.0
    
    @staticmethod
    def calculate_energy_efficiency(energy_consumed: float, tasks_completed: int) -> float:
        """Calculate energy efficiency (tasks per unit energy)"""
        return tasks_completed / energy_consumed if energy_consumed > 0 else 0.0
    
    @staticmethod
    def calculate_qos_satisfaction(satisfied_tasks: int, total_tasks: int) -> float:
        """Calculate QoS satisfaction rate"""
        return satisfied_tasks / total_tasks if total_tasks > 0 else 0.0