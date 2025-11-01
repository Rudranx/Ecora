import numpy as np
import networkx as nx
from typing import Dict, Tuple, List # <-- IMPORT LIST

class SocialNetworkModel:
    """Model social connections between vehicles"""
    
    def __init__(self, entity_ids: List[int]): # <-- MODIFIED
        self.entity_ids = entity_ids # <-- MODIFIED
        self.num_entities = len(entity_ids) # <-- MODIFIED
        self.social_graph = nx.Graph()
        self.interest_vectors = {}
        self.trust_scores = {}
        
        self._initialize_social_network()
        
    def _initialize_social_network(self):
        """Initialize social network with random connections"""
        # Add nodes
        for entity_id in self.entity_ids: # <-- MODIFIED
            self.social_graph.add_node(entity_id)
            # Random interest vector
            self.interest_vectors[entity_id] = np.random.rand(10)
            
        # Add edges (social connections) - preferential attachment
        for i in range(self.num_entities):
            v1 = self.entity_ids[i] # <-- MODIFIED
            num_connections = np.random.poisson(3)  # Average 3 connections
            for _ in range(num_connections):
                # Select a random entity ID
                v2 = self.entity_ids[np.random.randint(self.num_entities)] # <-- MODIFIED
                if v1 != v2:
                    # Weight represents relationship strength
                    weight = np.random.random()
                    self.social_graph.add_edge(v1, v2, weight=weight)
                    
    def calculate_interest_similarity(self, v1: int, v2: int) -> float:
        """Calculate cosine similarity between interest vectors"""
        vec1 = self.interest_vectors.get(v1, np.zeros(10))
        vec2 = self.interest_vectors.get(v2, np.zeros(10))
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 * norm2 > 0:
            return dot_product / (norm1 * norm2)
        return 0.0
    
    def calculate_social_trust(self, v1: int, v2: int) -> float:
        """Calculate trust based on shortest path length (Eq. 7/8 simplified)"""
        try:
            # Check if nodes exist and a path exists
            if (v1 in self.social_graph and 
                v2 in self.social_graph and 
                nx.has_path(self.social_graph, v1, v2)):
                
                path_length = nx.shortest_path_length(self.social_graph, v1, v2)
                # Trust decreases with path length
                trust = 1.0 / (1 + path_length)
            else:
                trust = 0.0
        except:
            trust = 0.0 # Fail safe
            
        return trust
    
    def get_social_connection(self, v1: int, v2: int, 
                            alpha: float = 0.5, beta: float = 0.5) -> float:
        """Get overall social connection strength (Eq. 9)"""
        similarity = self.calculate_interest_similarity(v1, v2)
        trust = self.calculate_social_trust(v1, v2)
        
        # Paper's Eq. 9
        return alpha * similarity + beta * trust