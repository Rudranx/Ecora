import matplotlib.pyplot as plt
import numpy as np
from simulation.simulator import ECORASimulator
import traceback
import copy

def run_experiment(config_file: str, sim_params: dict):
    """
    Runs a single simulation with specific parameters
    and returns the convergence history (in SECONDS).
    """
    print(f"--- Running experiment with params: {sim_params} ---")
    
    simulator = ECORASimulator(config_file, sim_params)
    simulator.initialize_network()
    
    # --- FIX for 0 tasks ---
    # We must run at least one trial that *has* tasks.
    history = None
    for i in range(10): # Try up to 10 times
        simulator.current_time = i
        history = simulator.run_time_slot()
        if history: # If tasks were generated, history is not empty
            break
            
    if not history:
        print("--- Experiment finished. No tasks generated, no history. ---")
        max_iter = simulator.config['algorithms']['cuckoo']['max_iterations']
        return [100.0] * (max_iter + 1) # Return 100s penalty
    # --- END FIX ---

    max_iter = simulator.config['algorithms']['cuckoo']['max_iterations']
    if len(history) < (max_iter + 1):
        last_val = history[-1]
        padding = [last_val] * ((max_iter + 1) - len(history))
        history.extend(padding)

    print(f"--- Experiment finished. Final delay: {history[-1] * 1000:.4f} ms ---")
    return history


def plot_figure(results: dict, title: str, filename: str, xlabel: str):
    """
    Generic plotting function for convergence graphs (Figs 2, 3, 4)
    """
    plt.figure(figsize=(10, 6))
    
    x_ticks_paper = np.arange(0, 26, 5)
    
    for label, history_s in results.items():
        if history_s:
            # --- CRITICAL FIX: Convert Seconds to Milliseconds for plotting ---
            history_ms = [s * 1000 for s in history_s]
            plt.plot(history_ms, label=f"ECORA ({label})", marker='o', markersize=4, linestyle='-') 
        
    plt.title(title)
    plt.xlabel('The number of iterations(times)')
    plt.ylabel('Average task processing latency (ms)') # Label is now correct
    plt.legend()
    plt.grid(True)
    plt.xticks(x_ticks_paper)
    plt.xlim(0, 25)
    plt.savefig(filename)
    print(f"\nSaved Figure to '{filename}'")
    plt.show()


if __name__ == "__main__":
    CONFIG_FILE = 'config.yaml'
    
    try:
        # ---
        # --- Figure 2: Varying Traffic Density ---
        # ---
        print("\n" + "="*50)
        print("RUNNING FIGURE 2 EXPERIMENT (TRAFFIC DENSITY)")
        print("="*50)
        traffic_densities = [0.06, 0.08, 0.10, 0.12]
        fig_2_results = {}
        
        for density in traffic_densities:
            params = {
                'task_generation_prob': density,
                'average_speed_kmh': 60 
            }
            history = run_experiment(CONFIG_FILE, params)
            fig_2_results[f"Density={density}"] = history
            
        plot_figure(
            fig_2_results,
            'Fig 2 (ECORA): Avg. Delay vs. Iterations (Varying Traffic Density)',
            'ecora_figure_2_traffic_density.png',
            'Traffic Density (as Task Gen. Prob.)'
        )

        # ---
        # --- Figure 3: Varying Average Speed ---
        # ---
        print("\n" + "="*50)
        print("RUNNING FIGURE 3 EXPERIMENT (AVERAGE SPEED)")
        print("="*50)
        average_speeds = [30, 60, 90, 120]
        fig_3_results = {}
        
        for speed in average_speeds:
            params = {
                'task_generation_prob': 0.10, # Use a constant density
                'average_speed_kmh': speed
            }
            history = run_experiment(CONFIG_FILE, params)
            fig_3_results[f"Speed={speed} km/h"] = history
            
        plot_figure(
            fig_3_results,
            'Fig 3 (ECORA): Avg. Delay vs. Iterations (Varying Average Speed)',
            'ecora_figure_3_average_speed.png',
            'Average Speed (km/h)'
        )

        # ---
        # --- Figure 4: Varying Algorithm Parameters (Pa and alpha3) ---
        # ---
        print("\n" + "="*50)
        print("RUNNING FIGURE 4a EXPERIMENT (Varying Pa)")
        print("="*50)
        pa_values = [0.05, 0.15, 0.25, 0.35, 0.45]
        fig_4a_results = {}
        
        base_params_fig4 = {
            'task_generation_prob': 0.10,
            'average_speed_kmh': 60
        }
        
        for pa in pa_values:
            params = copy.deepcopy(base_params_fig4)
            params['discovery_probability'] = pa 
            
            history = run_experiment(CONFIG_FILE, params)
            fig_4a_results[f"Pa={pa}"] = history
            
        plot_figure(
            fig_4a_results,
            'Fig 4a (ECORA): Avg. Delay vs. Iterations (Varying Pa)',
            'ecora_figure_4a_pa.png',
            'Parameter Pa (discovery_probability)'
        )
        
        print("\n" + "="*50)
        print("RUNNING FIGURE 4b EXPERIMENT (Varying alpha3)")
        print("="*50)
        alpha3_values = [0.01, 0.1, 1, 5, 10]
        fig_4b_results = {}
        
        for a3 in alpha3_values:
            params = copy.deepcopy(base_params_fig4)
            params['step_size_factor'] = a3 
            
            history = run_experiment(CONFIG_FILE, params)
            fig_4b_results[f"a3={a3}"] = history
            
        plot_figure(
            fig_4b_results,
            'Fig 4b (ECORA): Avg. Delay vs. Iterations (Varying a3)',
            'ecora_figure_4b_alpha3.png',
            'Parameter a3 (step_size_factor)'
        )
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        traceback.print_exc()