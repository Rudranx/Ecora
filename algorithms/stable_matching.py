"""
Corrected Implementation of Algorithm 1: Stable Matching Based on Mobile Social Contact
For service caching in vehicular social networks
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ServiceCacheRequest:
    """Data structure for service cache requests"""
    cv_id: int
    ap_id: int
    service_id: int
    preference_score: float

class StableMatchingAlgorithm:
    """
    Implements Algorithm 1: Stable Matching Algorithm Based on Mobile Social Contact
    This allocates service caches to collaborative vehicles based on social connections
    """
    
    def __init__(self, network, service_popularities, num_services):
        self.network = network
        self.service_popularities = service_popularities
        self.num_services = num_services
        self.matching_result: Dict[int, int] = {}
        
        # Initialize social model components
        self.cv_ids = [cv.id for cv in network.collaborative_vehicles]
        self.ap_ids = [ap.id for ap in network.access_points]
        
        # Initialize interest vectors for social similarity calculation
        self.interest_vectors = {}
        for entity_id in self.cv_ids + self.ap_ids:
            self.interest_vectors[entity_id] = np.random.rand(10)  # 10-dimensional interest vector
        
        # Cache constraints from paper
        self.alpha2 = 0.5  # Social connection weight
        self.beta2 = 0.5   # Trust weight
        
    def calculate_interest_similarity(self, entity1_id: int, entity2_id: int) -> float:
        """
        Calculate interest similarity S_m,y (Equation 6)
        Using cosine similarity between interest vectors
        """
        vec1 = self.interest_vectors.get(entity1_id, np.zeros(10))
        vec2 = self.interest_vectors.get(entity2_id, np.zeros(10))
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 * norm2 > 0:
            return dot_product / (norm1 * norm2)
        return 0.0
    
    def calculate_social_trust(self, entity1, entity2) -> float:
        """
        Calculate social trust B_m,y (Equations 7-8)
        Simplified using distance-based trust metric
        """
        distance = entity1.distance_to(entity2)
        max_distance = np.sqrt(self.network.area_size[0]**2 + self.network.area_size[1]**2)
        
        # Normalized trust based on proximity
        trust = 1.0 - (distance / max_distance)
        return max(0.0, trust)
    
    def calculate_social_connection(self, entity1, entity2) -> float:
        """
        Calculate overall social connection θ_k,n (Equation 9)
        θ_k,n = α2 * S_k,n + β2 * B_k,n
        """
        similarity = self.calculate_interest_similarity(entity1.id, entity2.id)
        trust = self.calculate_social_trust(entity1, entity2)
        
        return self.alpha2 * similarity + self.beta2 * trust
    
    def calculate_ap_preference(self, ap, cv) -> float:
        """
        Calculate AP's preference for CV (Equation 16)
        Y^AP_k,n = (θ_k,n * QoS_k,n) / t^up_k,n
        """
        # Social connection
        theta_kn = self.calculate_social_connection(ap, cv)
        
        # Check communication range
        distance = ap.distance_to(cv)
        if distance > min(ap.communication_range, cv.communication_range):
            return 0.0
        
        # QoS approximated by channel capacity
        tx_power_dbm = self.network.config['channel']['ap_tx_power_dbm']
        qos_kn = self.network.get_channel_capacity_mbps(distance, tx_power_dbm)
        if qos_kn <= 0:
            return 0.0
        
        # Transmission time (using reference size)
        transmission_time = 1.0 / (qos_kn + 0.01)
        
        # Calculate preference (Equation 16)
        preference = (theta_kn * qos_kn) / (transmission_time + 0.01)
        return preference
    
    def calculate_cv_preference(self, cv, ap) -> float:
        """
        Calculate CV's preference for AP (Equation 17)
        Y^CV_k,n = (D_k,n * QR_k,n) / (1 + e^(-ρ(f)_max))
        """
        # Check communication range
        distance = cv.distance_to(ap)
        if distance > min(ap.communication_range, cv.communication_range):
            return 0.0
        
        # Movement correlation D_k,n (Equation 10)
        direction_vec = ap.position - cv.position
        velocity_dot = np.dot(cv.velocity, direction_vec)
        mu = 1.0 if velocity_dot > 0 else 0.5  # Equation 11
        
        R_k = ap.communication_range
        D_kn = 1 - np.exp(-(mu * R_k) / (distance + 0.01))
        
        # Link quality QR_k,n
        tx_power_dbm = self.network.config['channel']['vehicle_tx_power_dbm']
        QR_kn = self.network.get_channel_capacity_mbps(distance, tx_power_dbm)
        if QR_kn <= 0:
            return 0.0
        
        # Maximum popularity of uncached services
        rho_f_max = np.max(self.service_popularities)
        
        # Calculate preference (Equation 17)
        preference = (D_kn * QR_kn) / (1 + np.exp(-rho_f_max))
        return preference
    
    def run(self) -> Dict[int, int]:
        """
        Execute Algorithm 1: Stable Matching for service caching
        Returns: Dictionary mapping CV_id -> AP_id for cache downloads
        """
        cvs = self.network.collaborative_vehicles
        aps = self.network.access_points
        
        if not cvs or not aps:
            return {}
        
        # Build preference lists
        ap_preferences = {}  # {ap_id: [cv_id_1, cv_id_2, ...]}
        cv_preferences = {}  # {cv_id: [ap_id_1, ap_id_2, ...]}
        
        # Calculate AP preferences over CVs
        for ap in aps:
            scores = []
            for cv in cvs:
                score = self.calculate_ap_preference(ap, cv)
                if score > 0:
                    scores.append((cv.id, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            ap_preferences[ap.id] = [cv_id for cv_id, _ in scores]
        
        # Calculate CV preferences over APs
        for cv in cvs:
            scores = []
            for ap in aps:
                score = self.calculate_cv_preference(cv, ap)
                if score > 0:
                    scores.append((ap.id, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            cv_preferences[cv.id] = [ap_id for ap_id, _ in scores]
        
        # Run Gale-Shapley algorithm (CVs propose to APs)
        unmatched_cvs = set(cv.id for cv in cvs)
        cv_next_proposal = {cv.id: 0 for cv in cvs}
        ap_current_matches = {ap.id: [] for ap in aps}
        ap_max_conns = {ap.id: ap.max_connections for ap in aps}
        self.matching_result.clear()
        
        # Pre-calculate AP rankings for efficiency
        ap_rankings = {}
        for ap_id, pref_list in ap_preferences.items():
            ap_rankings[ap_id] = {cv_id: rank for rank, cv_id in enumerate(pref_list)}
        
        # Main matching loop
        max_iterations = 100
        iteration = 0
        
        while unmatched_cvs and iteration < max_iterations:
            iteration += 1
            
            if not unmatched_cvs:
                break
                
            cv_id = unmatched_cvs.pop()
            
            # Check if CV has more proposals
            if cv_id not in cv_preferences or cv_next_proposal[cv_id] >= len(cv_preferences[cv_id]):
                continue
            
            # Get next AP to propose to
            ap_id = cv_preferences[cv_id][cv_next_proposal[cv_id]]
            cv_next_proposal[cv_id] += 1
            
            # Check if AP ranks this CV
            if cv_id not in ap_rankings.get(ap_id, {}):
                unmatched_cvs.add(cv_id)
                continue
            
            # Check AP's connection limit (Equation 3: Constraint C4)
            if len(ap_current_matches[ap_id]) < ap_max_conns[ap_id]:
                # AP has free slots
                ap_current_matches[ap_id].append(cv_id)
                self.matching_result[cv_id] = ap_id
            else:
                # AP is full, check if should replace
                current_matches = ap_current_matches[ap_id]
                worst_match = max(current_matches,
                                key=lambda x: ap_rankings[ap_id].get(x, float('inf')))
                
                # Compare rankings
                if ap_rankings[ap_id][cv_id] < ap_rankings[ap_id].get(worst_match, float('inf')):
                    # Replace worst match
                    ap_current_matches[ap_id].remove(worst_match)
                    ap_current_matches[ap_id].append(cv_id)
                    self.matching_result[cv_id] = ap_id
                    if worst_match in self.matching_result:
                        del self.matching_result[worst_match]
                    unmatched_cvs.add(worst_match)
                else:
                    # Reject proposal
                    unmatched_cvs.add(cv_id)
        
        # After matching, allocate services to matched CVs based on popularity
        self._allocate_services_to_cvs()
        
        return self.matching_result
    
    def _allocate_services_to_cvs(self):
        """
        Allocate services to matched CVs based on popularity and capacity constraints
        Implements the service caching part of Algorithm 1
        """
        # Sort services by popularity (highest first)
        service_order = np.argsort(self.service_popularities)[::-1]
        
        for cv in self.network.collaborative_vehicles:
            if cv.id not in self.matching_result:
                continue
                
            # Initialize CV's cached services
            cv.cached_services = set()
            cache_used = 0
            cache_capacity = cv.storage_resources * 0.5  # Use 50% for caching
            
            # Try to cache services in order of popularity
            for service_id in service_order:
                # Simplified: assume each service takes 20% of cache
                service_size = cache_capacity * 0.2
                
                # Check capacity constraint (Equation 2)
                if cache_used + service_size <= cache_capacity:
                    cv.cached_services.add(service_id)
                    cache_used += service_size
                    
                    # Cache at most 3 services per CV
                    if len(cv.cached_services) >= 3:
                        break
