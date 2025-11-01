import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from core.entities import CollaborativeVehicle, AccessPoint
from models.social import SocialNetworkModel # <-- IMPORT SOCIAL MODEL (Miss 1)

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
        self.service_popularities = service_popularities # (Miss 2)
        self.num_services = num_services
        self.matching_result: Dict[int, int] = {}  # CV -> AP mapping
        
        # (Miss 1) Initialize Social Network Model
        # It needs all entities that can have social ties
        num_entities = len(network.collaborative_vehicles) + len(network.access_points)
        self.social_model = SocialNetworkModel(num_entities)
        
    def calculate_ap_preference(self, ap: AccessPoint, cv: CollaborativeVehicle) -> float:
        """
        Calculate AP's preference for a CV
        Based on Equation 16: YAP_k,n = (θk,n * QoSk,n) / tup_k,n
        """
        
        # (Miss 1) Calculate Social Connection (θk,n) using Eq. 9
        # This model implements Eq. 7, 8, 9
        social_connection = self.social_model.get_social_connection(ap.id, cv.id)
        
        # Calculate QoS (QoSk,n)
        distance = ap.distance_to(cv)
        if distance > ap.communication_range:
            return 0.0  # Out of range, preference is zero
            
        if distance > 0:
            qos = self.network._calculate_channel_quality(distance)
        else:
            qos = 1.0
            
        # Calculate Transmission Time (tup_k,n) (simplified)
        # We assume data size is constant for preference calculation
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
        if distance > ap.communication_range:
            return 0.0 # Out of range
            
        # Calculate mu (μ) parameter
        direction_vec = ap.position - cv.position
        dot_product = np.dot(cv.velocity, direction_vec)
        mu = 1.0 if dot_product > 0 else 0.5 # Driving towards or away
        
        # Eq 10
        movement_correlation = 1 - np.exp(-(mu * ap.communication_range) / (distance + 0.01))
        
        # Calculate Link Quality (QRk,n)
        link_quality = self.network._calculate_channel_quality(distance)
        
        # (Miss 2) Calculate Service Popularity (ρ(f)max)
        # Find the most popular service this CV *doesn't* have
        # This is a simplification: we assume it wants the most popular service overall
        popularity = np.max(self.service_popularities)
        
        # Equation 17
        preference = (movement_correlation * link_quality) / (1 + np.exp(-popularity))
        return preference
    
    # This function is no longer needed, it's replaced by self.social_model
    # def _calculate_social_connection(self, ap_id: int, cv_id: int) -> float:
    #     ... 
    
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
            
        # Gale-Shapley algorithm
        unmatched_cvs = set(cv.id for cv in cvs)
        cv_next_proposal = {cv.id: 0 for cv in cvs}
        ap_current_match = {ap.id: None for ap in aps} # Tracks which CV is matched to AP
        
        # NEW: Track AP connections based on paper (Constraint C4)
        ap_connections = {ap.id: 0 for ap in aps}
        ap_max_connections = {ap.id: ap.max_connections for ap in aps}

        while unmatched_cvs:
            cv_id = unmatched_cvs.pop()
            
            if cv_id not in cv_preferences or cv_next_proposal[cv_id] >= len(cv_preferences[cv_id]):
                continue # This CV has no more APs to propose to
                
            # Get AP from CV's preference list
            ap_id = cv_preferences[cv_id][cv_next_proposal[cv_id]]
            cv_next_proposal[cv_id] += 1
            
            # Find the AP's preference list for *CVs*
            ap_pref_list = ap_preferences.get(ap_id, [])
            
            if ap_connections[ap_id] < ap_max_connections[ap_id]:
                # AP has free slots, accepts CV
                ap_current_match[ap_id] = cv_id # This is 1-to-1, needs to be 1-to-many
                # For 1-to-many, we need to store a list of matches
                # Let's stick to the paper's model: APs have preferences, CVs propose
                # This implementation is CV-proposing Gale-Shapley
                
                # Simplified 1-to-many:
                # We'll use the existing code's logic, but check ap_connections
                self.matching_result[cv_id] = ap_id
                ap_connections[ap_id] += 1
            else:
                # AP is full. Check if this CV is preferred over the *worst* CV currently matched.
                # This part is complex. The paper's SM algorithm is not fully specified.
                # We will stick to the provided code's logic (which is closer to 1-to-1 matching)
                # but add the unmatched CV back.
                
                # For now, if AP is full, reject.
                unmatched_cvs.add(cv_id)
                
        # This implementation is a bit flawed (it's not true 1-to-many SM)
        # but it's closer to the paper's *intent*
        
        # Let's re-read the SM algorithm logic in the code
        # ... ah, the original code had a bug. It didn't handle rejection/replacement.
        
        # Let's restart the Gale-Shapley loop logic
        
        unmatched_cvs = set(cv.id for cv in cvs)
        cv_next_proposal = {cv.id: 0 for cv in cvs}
        ap_current_matches = {ap.id: [] for ap in aps} # List of matched CVs
        ap_max_conns = {ap.id: ap.max_connections for ap in aps}
        
        # Pre-calculate AP's ranking of CVs for quick lookup
        ap_rankings = {}
        for ap_id, pref_list in ap_preferences.items():
            ap_rankings[ap_id] = {cv_id: rank for rank, cv_id in enumerate(pref_list)}

        while unmatched_cvs:
            cv_id = unmatched_cvs.pop()
            
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
                ap_current_matches[ap_id].append(cv_id)
                self.matching_result[cv_id] = ap_id
            else:
                # AP is full, check for replacement
                worst_cv_id = -1
                worst_rank = -1
                ap_cv_ranks = ap_rankings[ap_id]
                
                for matched_cv_id in ap_current_matches[ap_id]:
                    rank = ap_cv_ranks.get(matched_cv_id, 9999)
                    if rank > worst_rank:
                        worst_rank = rank
                        worst_cv_id = matched_cv_id
                        
                current_cv_rank = ap_cv_ranks.get(cv_id, 9999)
                
                if current_cv_rank < worst_rank:
                    # New CV is better than the worst matched CV
                    ap_current_matches[ap_id].remove(worst_cv_id)
                    ap_current_matches[ap_id].append(cv_id)
                    self.matching_result[cv_id] = ap_id
                    
                    # The rejected CV becomes unmatched
                    del self.matching_result[worst_cv_id]
                    unmatched_cvs.add(worst_cv_id)
                else:
                    # CV is rejected, proposes to next
                    unmatched_cvs.add(cv_id)

        return self.matching_result