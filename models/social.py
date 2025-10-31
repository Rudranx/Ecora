import numpy as np
import networkx as nx
from typing import Dict, Tuple

class SocialNetworkModel:
    """Model social connections between vehicles"""
    
    def __init__(self, num_vehicles: int):
        self.num_vehicles = num_vehicles
        self.social_graph = nx.Graph()
        self.interest_vectors = {}
        self.trust_scores = {}
        
        self._initialize_social_network()
        
    def _initialize_social_network(self):
        """Initialize social network with random connections"""
        # Add nodes
        for i in range(self.num_vehicles):
            self.social_graph.add_node(i)
            # Random interest vector
            self.interest_vectors[i] = np.random.rand(10)
            
        # Add edges (social connections) - preferential attachment
        for i in range(self.num_vehicles):
            num_connections = np.random.poisson(3)  # Average 3 connections
            for _ in range(num_connections):
                j = np.random.randint(self.num_vehicles)
                if i != j:
                    # Weight represents relationship strength
                    weight = np.random.random()
                    self.social_graph.add_edge(i, j, weight=weight)
                    
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
        """Calculate trust based on betweenness centrality"""
        try:
            # Check if path exists
            if nx.has_path(self.social_graph, v1, v2):
                path_length = nx.shortest_path_length(self.social_graph, v1, v2)
                # Trust decreases with path length
                trust = 1.0 / (1 + path_length)
            else:
                trust = 0.0
        except:
            trust = 0.0
            
        return trust
    
    def get_social_connection(self, v1: int, v2: int, 
                            alpha: float = 0.5, beta: float = 0.5) -> float:
        """Get overall social connection strength"""
        similarity = self.calculate_interest_similarity(v1, v2)
        trust = self.calculate_social_trust(v1, v2)
        
        return alpha * similarity + beta * trust