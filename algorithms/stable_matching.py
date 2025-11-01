import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from core.entities import CollaborativeVehicle, AccessPoint
from models.social import SocialNetworkModel # (No change here)

@dataclass
class ServiceCacheRequest:
    cv_id: int
    ap_id: int
    service_id: int
    preference_score: float

class StableMatchingAlgorithm:
    """Stable matching for service caching between APs and CVs"""
    
    def __init__(self, network, service_popularities, num_services):
        self.network = network
        self.service_popularities = service_popularities
        self.num_services = num_services
        self.matching_result: Dict[int, int] = {}  # CV -> AP mapping
        
        # --- THIS IS THE FIX ---
        # Get the *actual* IDs of all entities
        cv_ids = [cv.id for cv in network.collaborative_vehicles]
        ap_ids = [ap.id for ap in network.access_points]
        all_entity_ids = cv_ids + ap_ids
        
        # Initialize the Social Model with the *correct* IDs
        self.social_model = SocialNetworkModel(all_entity_ids)
        # --- END FIX ---
        
    def calculate_ap_preference(self, ap: AccessPoint, cv: CollaborativeVehicle) -> float:
        """
        Calculate AP's preference for a CV
        Based on Equation 16: YAP_k,n = (θk,n * QoSk,n) / tup_k,n
        """
        
        # (Miss 1) Calculate Social Connection (θk,n) using Eq. 9
        # This will now work correctly
        social_connection = self.social_model.get_social_connection(ap.id, cv.id)
        
        # Calculate QoS (QoSk,n)
        distance = ap.distance_to(cv)
        
        # --- SECONDARY FIX: Check *both* communication ranges ---
        link_range = min(ap.communication_range, cv.communication_range)
        if distance > link_range:
            return 0.0  # Out of range, preference is zero
            
        if distance > 0:
            tx_power_dbm = self.network.config['ap_tx_power_dbm']
            qos = self.network.get_channel_capacity_mbps(distance, tx_power_dbm)
        else:
            qos = 1000.0 # Very high capacity at 0 distance
            
        # Calculate Transmission Time (tup_k,n) (simplified)
        transmission_time = 1.0 / (qos + 0.01) 
        
        # Equation 16
        preference = (social_connection * qos) / (transmission_time + 0.01)
        return preference
    
    def calculate_cv_preference(self, cv: CollaborativeVehicle, ap: AccessPoint) -> float:
        """
        Calculate CV's preference for an AP
        Based on Equation 17: YCV_k,n = (Dk,n * QRk,n) / (1 + e^(-ρ(f)max))
        """
        
        # (Miss 3) Calculate Movement Correlation (Dk,n) using Eq. 10
        distance = ap.distance_to(cv)
        
        # --- SECONDARY FIX: Check *both* communication ranges ---
        link_range = min(ap.communication_range, cv.communication_range)
        if distance > link_range:
            return 0.0 # Out of range
            
        # Calculate mu (μ) parameter
        direction_vec = ap.position - cv.position
        dot_product = np.dot(cv.velocity, direction_vec)
        mu = 1.0 if dot_product > 0 else 0.5 # Driving towards or away
        
        # Eq 10
        movement_correlation = 1 - np.exp(-(mu * ap.communication_range) / (distance + 0.01))
        
        # Calculate Link Quality (QRk,n)
        if distance > 0:
            tx_power_dbm = self.network.config['vehicle_tx_power_dbm']
            link_quality = self.network.get_channel_capacity_mbps(distance, tx_power_dbm)
        else:
            link_quality = 1000.0 # Very high capacity
        
        # (Miss 2) Calculate Service Popularity (ρ(f)max)
        popularity = np.max(self.service_popularities)
        
        # Equation 17
        preference = (movement_correlation * link_quality) / (1 + np.exp(-popularity))
        return preference
    
    def run(self) -> Dict[int, int]:
        """Execute stable matching algorithm"""
        cvs = self.network.collaborative_vehicles
        aps = self.network.access_points
        
        if not cvs or not aps:
            return {}
            
        # Build preference lists
        ap_preferences = {}
        cv_preferences = {}
        
        for ap in aps:
            preferences = []
            for cv in cvs:
                score = self.calculate_ap_preference(ap, cv)
                if score > 0:
                    preferences.append((cv.id, score))
            preferences.sort(key=lambda x: x[1], reverse=True)
            ap_preferences[ap.id] = [cv_id for cv_id, _ in preferences]
            
        for cv in cvs:
            preferences = []
            for ap in aps:
                score = self.calculate_cv_preference(cv, ap)
                if score > 0:
                    preferences.append((ap.id, score))
            preferences.sort(key=lambda x: x[1], reverse=True)
            cv_preferences[cv.id] = [ap_id for ap_id, _ in preferences]
            
        # Gale-Shapley algorithm (CVs propose to APs)
        unmatched_cvs = set(cv.id for cv in cvs)
        cv_next_proposal = {cv.id: 0 for cv in cvs}
        ap_current_matches = {ap.id: [] for ap in aps} # List of matched CVs
        ap_max_conns = {ap.id: ap.max_connections for ap in aps}
        self.matching_result.clear()
        
        # Pre-calculate AP's ranking of CVs for quick lookup
        ap_rankings = {}
        for ap_id, pref_list in ap_preferences.items():
            ap_rankings[ap_id] = {cv_id: rank for rank, cv_id in enumerate(pref_list)}

        while unmatched_cvs:
            try:
                cv_id = unmatched_cvs.pop()
            except KeyError:
                break # All CVs are matched

            if cv_id not in cv_preferences or cv_next_proposal[cv_id] >= len(cv_preferences[cv_id]):
                continue # No more proposals for this CV

            ap_id = cv_preferences[cv_id][cv_next_proposal[cv_id]]
            cv_next_proposal[cv_id] += 1

            if cv_id not in ap_rankings.get(ap_id, {}):
                # AP doesn't want this CV (or CV is out of range), CV proposes to next
                unmatched_cvs.add(cv_id)
                continue

            if len(ap_current_matches[ap_id]) < ap_max_conns[ap_id]:
                # AP has a free slot
                ap_current_matches[ap.id].append(cv_id)
                self.matching_result[cv_id] = ap_id
            else:
                # AP is full, check for replacement
                worst_cv_id = -1
                worst_rank = -1
                ap_cv_ranks = ap_rankings.get(ap_id, {}) # Use .get for safety
                if not ap_cv_ranks:
                    unmatched_cvs.add(cv_id)
                    continue

                for matched_cv_id in ap_current_matches[ap_id]:
                    rank = ap_cv_ranks.get(matched_cv_id, 9999) # 9999 = very bad rank
                    if rank > worst_rank:
                        worst_rank = rank
                        worst_cv_id = matched_cv_id
                        
                current_cv_rank = ap_cv_ranks.get(cv_id, 9999)
                
                if current_cv_rank < worst_rank:
                    # New CV is better than the worst matched CV
                    ap_current_matches[ap.id].remove(worst_cv_id)
                    ap_current_matches[ap.id].append(cv_id)
                    self.matching_result[cv_id] = ap_id
                    
                    if worst_cv_id in self.matching_result:
                        del self.matching_result[worst_cv_id]
                    unmatched_cvs.add(worst_cv_id)
                else:
                    # CV is rejected, proposes to next
                    unmatched_cvs.add(cv_id)

        return self.matching_result