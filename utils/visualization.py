import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

class Visualizer:
    """Visualization utilities for ECORA simulation"""
    
    @staticmethod
    def plot_network_topology(network, save_path: str = None):
        """Plot network topology with vehicles and APs"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot mission vehicles
        for mv in network.mission_vehicles:
            ax.scatter(mv.position[0], mv.position[1], 
                      c='red', marker='o', s=50, label='Mission Vehicle')
            
        # Plot collaborative vehicles
        for cv in network.collaborative_vehicles:
            ax.scatter(cv.position[0], cv.position[1],
                      c='blue', marker='^', s=50, label='Collaborative Vehicle')
            
        # Plot access points
        for ap in network.access_points:
            ax.scatter(ap.position[0], ap.position[1],
                      c='green', marker='s', s=100, label='Access Point')
            # Draw communication range
            circle = plt.Circle(ap.position, ap.communication_range,
                               fill=False, linestyle='--', color='green', alpha=0.3)
            ax.add_patch(circle)
            
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) 
                 if l not in labels[:i]]
        ax.legend(*zip(*unique))
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Vehicular Network Topology')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    @staticmethod
    def plot_metrics_comparison(results_dict: Dict[str, Dict], 
                               save_path: str = None):
        """Compare metrics from multiple algorithms"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['average_delay', 'load_balance', 
                  'cache_hit_rate', 'offload_success_rate']
        titles = ['Average Delay (ms)', 'Load Balance',
                 'Cache Hit Rate', 'Offload Success Rate']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            for algo_name, results in results_dict.items():
                if metric in results:
                    ax.plot(results[metric], label=algo_name)
                    
            ax.set_title(title)
            ax.set_xlabel('Time Slot')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()