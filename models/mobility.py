import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

class MobilityModel(ABC):
    """Abstract base class for mobility models"""
    
    @abstractmethod
    def update_position(self, current_position: np.ndarray, 
                       velocity: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        pass

class RandomWaypointModel(MobilityModel):
    """Random Waypoint Mobility Model"""
    
    def __init__(self, area_size: Tuple[float, float], 
                 speed_range: Tuple[float, float]):
        self.area_size = area_size
        self.speed_range = speed_range
        self.waypoint = None
        self.pause_time = 0
        
    def update_position(self, current_position: np.ndarray,
                       velocity: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Update position based on random waypoint model"""
        
        if self.waypoint is None or np.linalg.norm(current_position - self.waypoint) < 1.0:
            # Generate new waypoint
            self.waypoint = np.random.rand(2) * self.area_size
            
            # Calculate new velocity towards waypoint
            direction = self.waypoint - current_position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                speed = np.random.uniform(*self.speed_range)
                velocity = (direction / distance) * speed
            else:
                velocity = np.zeros(2)
                
        # Update position
        new_position = current_position + velocity * dt
        
        # Boundary check
        new_position = np.clip(new_position, 0, self.area_size)
        
        return new_position, velocity

class ManhattanModel(MobilityModel):
    """Manhattan Grid Mobility Model"""
    
    def __init__(self, area_size: Tuple[float, float],
                 speed_range: Tuple[float, float], 
                 block_size: float = 100):
        self.area_size = area_size
        self.speed_range = speed_range
        self.block_size = block_size
        self.direction = None
        self.next_intersection = None
        
    def update_position(self, current_position: np.ndarray,
                       velocity: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Update position following Manhattan grid pattern"""
        
        if self.next_intersection is None:
            # Find nearest intersection
            grid_x = round(current_position[0] / self.block_size) * self.block_size
            grid_y = round(current_position[1] / self.block_size) * self.block_size
            self.next_intersection = np.array([grid_x, grid_y])
            
            # Choose random direction (N, S, E, W)
            directions = [
                np.array([0, 1]),   # North
                np.array([0, -1]),  # South
                np.array([1, 0]),   # East
                np.array([-1, 0])   # West
            ]
            self.direction = directions[np.random.randint(4)]
            
        # Move along grid
        speed = np.random.uniform(*self.speed_range)
        velocity = self.direction * speed
        new_position = current_position + velocity * dt
        
        # Check if reached intersection
        if np.linalg.norm(new_position - self.next_intersection) < 5.0:
            self.next_intersection = None
            
        # Boundary check
        new_position = np.clip(new_position, 0, self.area_size)
        
        return new_position, velocity