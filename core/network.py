import numpy as np
from typing import List, Dict, Tuple, Set
from core.entities import Vehicle, AccessPoint, MissionVehicle, CollaborativeVehicle

class VehicularNetwork:
    """Manages network topology and connectivity"""
    
    def __init__(self, area_size: Tuple[float, float]):
        self.area_size = area_size
        self.mission_vehicles: List[MissionVehicle] = []
        self.collaborative_vehicles: List[CollaborativeVehicle] = []
        self.access_points: List[AccessPoint] = []
        self.connectivity_matrix: Dict[Tuple[int, int], float] = {}
        
    def add_mission_vehicle(self, vehicle: MissionVehicle):
        self.mission_vehicles.append(vehicle)
        
    def add_collaborative_vehicle(self, vehicle: CollaborativeVehicle):
        self.collaborative_vehicles.append(vehicle)
        
    def add_access_point(self, ap: AccessPoint):
        self.access_points.append(ap)
        
    def update_connectivity(self):
        """Update connectivity matrix based on current positions"""
        self.connectivity_matrix.clear()
        
        # V2V connectivity
        for mv in self.mission_vehicles:
            for cv in self.collaborative_vehicles:
                distance = mv.distance_to(cv)
                if distance <= min(mv.communication_range, cv.communication_range):
                    # Calculate channel quality (simplified)
                    channel_quality = self._calculate_channel_quality(distance)
                    self.connectivity_matrix[(mv.id, cv.id)] = channel_quality
                    
        # V2I connectivity
        for mv in self.mission_vehicles:
            for ap in self.access_points:
                distance = ap.distance_to(mv)
                if distance <= min(mv.communication_range, ap.communication_range):
                    channel_quality = self._calculate_channel_quality(distance)
                    self.connectivity_matrix[(mv.id, ap.id)] = channel_quality
                    
    def _calculate_channel_quality(self, distance: float) -> float:
        """Calculate channel quality based on distance (simplified model)"""
        # Simple path loss model
        if distance < 1:
            distance = 1
        path_loss = 20 * np.log10(distance) + 20 * np.log10(2.4e9) - 147.55
        # Convert to quality metric (0-1)
        quality = max(0, 1 - path_loss / 100)
        return quality
    
    def get_neighbors(self, vehicle_id: int) -> List[int]:
        """Get all neighbors of a vehicle"""
        neighbors = []
        for (v1, v2), quality in self.connectivity_matrix.items():
            if v1 == vehicle_id:
                neighbors.append(v2)
            elif v2 == vehicle_id:
                neighbors.append(v1)
        return neighbors