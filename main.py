import matplotlib.pyplot as plt
import numpy as np
from simulation.simulator import ECORASimulator
import traceback

def plot_results(results):
    """Plot simulation results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Average delay
    if results['average_delay']:
        # Filter out inf values
        filtered_delays = [d for d in results['average_delay'] if d != float('inf') and d < 100]
        if filtered_delays:
            axes[0, 0].plot(filtered_delays)
    axes[0, 0].set_title('Average Task Processing Delay')
    axes[0, 0].set_xlabel('Time Slot')
    axes[0, 0].set_ylabel('Delay (ms)')
    axes[0, 0].grid(True)
    
    # Load balance
    axes[0, 1].plot(results['load_balance'])
    axes[0, 1].set_title('Load Balance (Variance)')
    axes[0, 1].set_xlabel('Time Slot')
    axes[0, 1].set_ylabel('Load Variance')
    axes[0, 1].grid(True)
    
    # Cache hit rate
    axes[1, 0].plot(results['cache_hit_rate'])
    axes[1, 0].set_title('Cache Hit Rate')
    axes[1, 0].set_xlabel('Time Slot')
    axes[1, 0].set_ylabel('Hit Rate')
    axes[1, 0].grid(True)
    
    # Offload success rate
    axes[1, 1].plot(results['offload_success_rate'])
    axes[1, 1].set_title('Offload Success Rate')
    axes[1, 1].set_xlabel('Time Slot')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('ecora_results.png')
    plt.show()
    
def print_statistics(results):
    """Print summary statistics"""
    print("\n" + "="*50)
    print("ECORA SIMULATION RESULTS SUMMARY")
    print("="*50)
    
    if results['average_delay']:
        # Filter out inf values
        filtered_delays = [d for d in results['average_delay'] if d != float('inf') and d < 100]
        if filtered_delays:
            print(f"Average Task Processing Delay: {np.mean(filtered_delays):.4f} ms")
            print(f"Min Delay: {np.min(filtered_delays):.4f} ms")
            print(f"Max Delay: {np.max(filtered_delays):.4f} ms")
        else:
            print("No valid delay measurements")
        
    if results['load_balance']:
        print(f"\nLoad Balance Variance: {np.mean(results['load_balance']):.4f}")
        
    if results['cache_hit_rate']:
        print(f"\nAverage Cache Hit Rate: {np.mean(results['cache_hit_rate']):.2%}")
        
    if results['offload_success_rate']:
        print(f"Average Offload Success Rate: {np.mean(results['offload_success_rate']):.2%}")
        
    print("="*50)

if __name__ == "__main__":
    try:
        # Run simulation
        simulator = ECORASimulator('config.yaml')
        results = simulator.run()
        
        # Display results
        print_statistics(results)
        plot_results(results)
    except Exception as e:
        print(f"Error during simulation: {e}")
        traceback.print_exc()