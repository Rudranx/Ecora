import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from core.entities import CollaborativeVehicle, AccessPoint

@dataclass
class ServiceCacheRequest:
    cv_id: int
    ap_id: int
    service_id: int
    preference_score: float

class StableMatchingAlgorithm:
    """Stable matching for service caching between APs and CVs"""
    
    def __init__(self, network, services):
        self.network = network
        self.services = services
        self.matching_result: Dict[int, int] = {}  # CV -> AP mapping
        
    def calculate_ap_preference(self, ap: AccessPoint, cv: CollaborativeVehicle) -> float:
        """Calculate AP's preference for a CV"""
        # Social connection
        social_connection = self._calculate_social_connection(ap.id, cv.id)
        
        # QoS
        distance = ap.distance_to(cv)
        if distance > 0:
            qos = self.network._calculate_channel_quality(distance)
        else:
            qos = 1.0
            
        # Transmission time (simplified)
        transmission_time = 1.0 / (qos + 0.01)
        
        preference = (social_connection * qos) / transmission_time
        return preference
    
    def calculate_cv_preference(self, cv: CollaborativeVehicle, ap: AccessPoint) -> float:
        """Calculate CV's preference for an AP"""
        # Movement correlation
        distance = ap.distance_to(cv)
        movement_correlation = 1 - np.exp(-ap.communication_range / (distance + 0.01))
        
        # Link quality
        link_quality = self.network._calculate_channel_quality(distance)
        
        # Service popularity (simplified - using random for now)
        popularity = np.random.random()
        
        preference = movement_correlation * link_quality / (1 + np.exp(-popularity))
        return preference
    
    def _calculate_social_connection(self, ap_id: int, cv_id: int) -> float:
        """Calculate social connection between AP and CV"""
        # Simplified social connection model
        # In real implementation, this would consider interest similarity and trust
        return 0.5 + 0.5 * np.random.random()
    
    def run(self) -> Dict[int, int]:
        """Execute stable matching algorithm"""
        cvs = self.network.collaborative_vehicles
        aps = self.network.access_points
        
        # Build preference lists
        ap_preferences = {}
        cv_preferences = {}
        
        for ap in aps:
            preferences = []
            for cv in cvs:
                score = self.calculate_ap_preference(ap, cv)
                preferences.append((cv.id, score))
            preferences.sort(key=lambda x: x[1], reverse=True)
            ap_preferences[ap.id] = [cv_id for cv_id, _ in preferences]
            
        for cv in cvs:
            preferences = []
            for ap in aps:
                score = self.calculate_cv_preference(cv, ap)
                preferences.append((ap.id, score))
            preferences.sort(key=lambda x: x[1], reverse=True)
            cv_preferences[cv.id] = [ap_id for ap_id, _ in preferences]
            
        # Gale-Shapley algorithm
        unmatched_cvs = set(cv.id for cv in cvs)
        cv_next_proposal = {cv.id: 0 for cv in cvs}
        ap_current_match = {ap.id: None for ap in aps}
        ap_connections = {ap.id: 0 for ap in aps}
        
        while unmatched_cvs and any(cv_next_proposal[cv] < len(cv_preferences[cv]) 
                                   for cv in unmatched_cvs):
            cv_id = unmatched_cvs.pop()
            
            if cv_next_proposal[cv_id] >= len(cv_preferences[cv_id]):
                continue
                
            ap_id = cv_preferences[cv_id][cv_next_proposal[cv_id]]
            cv_next_proposal[cv_id] += 1
            
            # Find AP by ID
            ap = next((ap for ap in aps if ap.id == ap_id), None)
            if not ap:
                unmatched_cvs.add(cv_id)
                continue
                
            if ap_connections[ap_id] < ap.max_connections:
                # AP accepts CV
                self.matching_result[cv_id] = ap_id
                ap_connections[ap_id] += 1
            else:
                # AP is full, check if should replace
                unmatched_cvs.add(cv_id)
                
        return self.matching_result