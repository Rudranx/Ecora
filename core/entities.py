import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class Vehicle:
    """Base class for vehicles"""
    id: int
    position: np.ndarray
    velocity: np.ndarray
    communication_range: float
    computing_resources: float
    storage_resources: float
    
    def update_position(self, dt: float):
        """Update vehicle position based on velocity"""
        self.position += self.velocity * dt
        
    def distance_to(self, other: 'Vehicle') -> float:
        """Calculate Euclidean distance to another vehicle"""
        return np.linalg.norm(self.position - other.position)

@dataclass
class MissionVehicle(Vehicle):
    """Mission vehicle that generates tasks"""
    current_task: Optional['Task'] = None
    offloading_target: Optional[int] = None
    
@dataclass
class CollaborativeVehicle(Vehicle):
    """Collaborative vehicle that provides resources"""
    cached_services: List[int] = field(default_factory=list)
    idle_computing_resources: float = 0.0
    idle_storage_resources: float = 0.0
    
    def can_cache_service(self, service_size: float) -> bool:
        """Check if vehicle can cache a service"""
        return self.idle_storage_resources >= service_size

@dataclass
class AccessPoint:
    """Access point (RSU + Edge Server)"""
    id: int
    position: np.ndarray
    communication_range: float
    computing_resources: float
    storage_resources: float
    max_connections: int
    current_connections: int = 0
    cached_services: List[int] = field(default_factory=list)
    
    def distance_to(self, vehicle: Vehicle) -> float:
        """Calculate distance to a vehicle"""
        return np.linalg.norm(self.position - vehicle.position)
    
    def can_accept_connection(self) -> bool:
        """Check if AP can accept more connections"""
        return self.current_connections < self.max_connections

@dataclass
class Task:
    """Computation task"""
    id: int
    data_size: float  # MB
    computation_demand: float  # GHz
    required_service: int
    generation_time: float
    deadline: float
    source_vehicle: int