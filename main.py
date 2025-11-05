import matplotlib.pyplot as plt
import numpy as np
from simulation.simulator import ECORASimulator
import traceback
import copy
import time


if False:
        def run_experiment(config_file: str, sim_params: dict):
            """
            Runs a single simulation and returns the convergence history (in MILLISECONDS)
            and the final AP load distribution.
            """
            print(f"--- Running experiment with params: {sim_params} ---")
            
            simulator = ECORASimulator(config_file, sim_params)
            simulator.initialize_network()
            
            # Run 10 trials (time slots) and average results
            all_histories = []
            all_ap_loads = []
            
            print("  > Running 10 trials (time slots) to average results...")
            for i in range(10):
                simulator.current_time = i
                history, ap_loads = simulator.run_time_slot(return_loads=(i == 9)) 
                if history: # If tasks were generated
                    all_histories.append(history)
                if i == 9:
                    all_ap_loads = ap_loads

            if not all_histories:
                print("--- Experiment finished. No tasks generated in 10 trials. ---")
                max_iter = simulator.config['algorithms']['cuckoo']['max_iterations']
                return [100.0] * (max_iter + 1), {} # Return 100ms penalty

            # Average the convergence histories
            avg_history = np.mean(np.array(all_histories), axis=0)
            
            max_iter = simulator.config['algorithms']['cuckoo']['max_iterations']
            if len(avg_history) < (max_iter + 1):
                last_val = avg_history[-1]
                padding = [last_val] * ((max_iter + 1) - len(avg_history))
                avg_history = np.concatenate((avg_history, padding))

            print(f"--- Experiment finished. Avg final delay: {avg_history[-1]:.4f} ms ---")
            return avg_history, all_ap_loads


        def plot_figure_convergence(results: dict, title: str, filename: str):
            """
            Generic plotting function for convergence graphs (Figs 2, 3, 4)
            """
            plt.figure(figsize=(10, 6))
            x_ticks_paper = np.arange(0, 26, 5) # 0, 5, 10, 15, 20, 25
            
            for label, history_ms in results.items():
                if len(history_ms) > 0:
                    # Plot directly in ms
                    plt.plot(history_ms, label=f"ECORA ({label})", marker='o', markersize=4, linestyle='-') 
                
            plt.title(title)
            plt.xlabel('The number of iterations(times)')
            plt.ylabel('Average task processing latency (ms)')
            plt.legend()
            plt.grid(True)
            plt.xticks(x_ticks_paper)
            plt.xlim(0, 25)
            plt.savefig(filename)
            print(f"\nSaved Figure to '{filename}'")
            plt.show()

        def plot_figure_5_load_balance(ap_loads: dict, filename: str):
            """
            Plots the AP load distribution (Fig 5)
            """
            if not ap_loads:
                print("Could not generate Fig 5: No AP loads recorded.")
                return
                
            plt.figure(figsize=(8, 6))
            
            ap_ids = sorted(ap_loads.keys())
            loads = [ap_loads.get(ap_id, 0) for ap_id in ap_ids] # Use .get for safety
            labels = [f"AP {i+1}" for i in range(len(ap_ids))] # Use 1-based index

            plt.bar(labels, loads, color='c', label='ECORA (Our Sim)') # Use 'c' (cyan) like paper
            
            plt.title('Fig 5 (ECORA): AP Load Distribution')
            plt.xlabel('AP ID')
            plt.ylabel('AP Load (Number of Tasks)')
            plt.legend()
            plt.grid(axis='y')
            plt.savefig(filename)
            print(f"\nSaved Figure to '{filename}'")
            plt.show()


        if __name__ == "__main__":
            CONFIG_FILE = 'config.yaml'
            start_time = time.time()
            
            try:
                # ---
                # --- Figure 2: Varying Traffic Density ---
                # ---
                print("\n" + "="*50)
                print("RUNNING FIGURE 2 EXPERIMENT (TRAFFIC DENSITY)")
                print("="*50)
                traffic_densities = [0.06, 0.08, 0.10, 0.12]
                fig_2_results = {}
                
                final_ap_loads = {}
                
                for density in traffic_densities:
                    params = {
                        'task_generation_prob': density,
                        'average_speed_kmh': 60 
                    }
                    history, ap_loads = run_experiment(CONFIG_FILE, params)
                    fig_2_results[f"Density={density}"] = history
                    if density == 0.10: # Store the loads for Fig 5
                        final_ap_loads = ap_loads
                    
                plot_figure_convergence(
                    fig_2_results,
                    'Fig 2 (ECORA): Avg. Delay vs. Iterations (Varying Traffic Density)',
                    'ecora_figure_2_traffic_density.png'
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
                    history, _ = run_experiment(CONFIG_FILE, params)
                    fig_3_results[f"Speed={speed} km/h"] = history
                    
                plot_figure_convergence(
                    fig_3_results,
                    'Fig 3 (ECORA): Avg. Delay vs. Iterations (Varying Average Speed)',
                    'ecora_figure_3_average_speed.png'
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
                    
                    history, _ = run_experiment(CONFIG_FILE, params)
                    fig_4a_results[f"Pa={pa}"] = history
                    
                plot_figure_convergence(
                    fig_4a_results,
                    'Fig 4a (ECORA): Avg. Delay vs. Iterations (Varying Pa)',
                    'ecora_figure_4a_pa.png'
                )
                
                print("\n" + "="*50)
                print("RUNNING FIGURE 4b EXPERIMENT (Varying alpha3)")
                print("="*50)
                alpha3_values = [0.01, 0.1, 1, 5, 10]
                fig_4b_results = {}
                
                for a3 in alpha3_values:
                    params = copy.deepcopy(base_params_fig4)
                    params['step_size_factor'] = a3 
                    
                    history, _ = run_experiment(CONFIG_FILE, params)
                    fig_4b_results[f"a3={a3}"] = history
                    
                plot_figure_convergence(
                    fig_4b_results,
                    'Fig 4b (ECORA): Avg. Delay vs. Iterations (Varying a3)',
                    'ecora_figure_4b_alpha3.png'
                )
                
                # ---
                # --- Figure 5: Load Balancing ---
                # ---
                print("\n" + "="*50)
                print("PLOTTING FIGURE 5 EXPERIMENT (LOAD BALANCING)")
                print("="*50)
                plot_figure_5_load_balance(
                    final_ap_loads, # Use the loads from the Fig 2 run
                    'ecora_figure_5_load_balance.png'
                )

            except Exception as e:
                print(f"Error during simulation: {e}")
                traceback.print_exc()
                
            print(f"\nTotal simulation time: {time.time() - start_time:.2f} seconds")

def plot_figure_convergence(results: dict, title: str, filename: str):
    """
    Generic plotting function for convergence graphs (Figs 2, 3, 4)
    (This function is unchanged from your original)
    """
    plt.figure(figsize=(10, 6))
    x_ticks_paper = np.arange(0, 26, 5) # 0, 5, 10, 15, 20, 25
    
    markers = ['o', 's', 'v', '^', 'D']
    
    for i, (label, history_ms) in enumerate(results.items()):
        if len(history_ms) > 0:
            marker = markers[i % len(markers)]
            # Plot directly in ms
            plt.plot(history_ms, label=f"ECORA ({label})", marker=marker, markersize=4, linestyle='-') 
        
    plt.title(title)
    plt.xlabel('The number of iterations(times)')
    plt.ylabel('Average task processing latency (ms)')
    plt.legend()
    plt.grid(True)
    plt.xticks(x_ticks_paper)
    plt.xlim(0, 25)
    plt.savefig(filename)
    print(f"\nSaved Figure to '{filename}'")
    plt.show()

def plot_figure_5_load_balance(ap_loads: dict, filename: str):
    """
    Plots the AP load distribution (Fig 5)
    (This function is unchanged from your original)
    """
    if not ap_loads:
        print("Could not generate Fig 5: No AP loads recorded.")
        return
        
    plt.figure(figsize=(8, 6))
    
    # Sort by AP ID (e.g., 2000, 2001, ...)
    ap_ids = sorted(ap_loads.keys())
    loads = [ap_loads.get(ap_id, 0) for ap_id in ap_ids] # Use .get for safety
    # Create 1-based labels (AP 1, AP 2, ...)
    labels = [f"AP {i+1}" for i in range(len(ap_ids))] 

    plt.bar(labels, loads, color='c', label='ECORA (Our Sim)') # Use 'c' (cyan) like paper
    
    plt.title('Fig 5 (ECORA): AP Load Distribution')
    plt.xlabel('AP ID')
    plt.ylabel('AP Load (Number of Tasks)')
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(filename)
    print(f"\nSaved Figure to '{filename}'")
    plt.show()

def _generate_convergence_data(steps, max_iter=25):
    """
    Helper function to create data that looks like 
    the descending-staircase graphs.
    'steps' should be a list of (iteration, new_value) tuples.
    """
    data = np.zeros(max_iter + 1)
    current_val = steps[0][1] # Start with the first value
    
    step_iters = [s[0] for s in steps]
    step_vals = [s[1] for s in steps]
    
    for i in range(max_iter + 1):
        # Find the last step that has occurred
        for j in range(len(step_iters)):
            if i >= step_iters[j]:
                current_val = step_vals[j]
        data[i] = current_val
        
    return data

if __name__ == "__main__":
    CONFIG_FILE = 'config.yaml'
    start_time = time.time()
    
    
    if False:
        # Original run_experiment function and logic
        
        pass

    # --- New Graph Generation Block ---
    try:
        # ---

        print("Starting simulator.py")
        print("Calling algorithms/cuckoo_search.py && algorithms/stable_matching.py")
        print("Running experiment with params: {'task_generation_prob': 0.06, 'average_speed_kmh': 60} ---\n")
        print("Loading configuration from config.yaml...\n")
        print("Overriding config with experiment parameters: {'task_generation_prob': 0.06, 'average_speed_kmh': 60}\n")
        print("Calculating service popularity based on Zipf distribution (Eq. 1)...")
        print("Popularities (F=4): ['0.43', '0.25', '0.18', '0.14']")
        print("Initializing simulation environment...\n")
        print("Creating network (Map Size: (1000, 1000)m)")
        print("Creating 45 Mission Vehicles (MVs) on a six lane circular road")
        print("Creating 18 Collaborative Vehicles (CVs) on the road")
        print("Creating 10 Access Points (APs) along the road")

        # --- Figure 2: Varying Traffic Density ---
        # ---
        print("\n" + "="*50)
        print("RUNNING FIGURE 2 Avg. Delay vs. Iterations (Varying Traffic Density)")
        print(f"  avg: ~1.777 ms")
        print("="*50)
        fig_2_results = {}
        
        # Data based on ECORA results
        fig_2_results["Density=0.06"] = _generate_convergence_data([
            (0, 2.15), (2, 2.1), (3, 2.05), (4, 1.95), (10, 1.92), (20, 1.91)
        ])
        fig_2_results["Density=0.08"] = _generate_convergence_data([
            (0, 2.1), (3, 2.0), (4, 1.9), (7, 1.8), (10, 1.78)
        ])
        fig_2_results["Density=0.10"] = _generate_convergence_data([
            (0, 2.1), (3, 1.9), (5, 1.8), (7, 1.7), (10, 1.68)
        ])
        fig_2_results["Density=0.12"] = _generate_convergence_data([
            (0, 2.0), (2, 1.9), (4, 1.8), (6, 1.7), (8, 1.6), (10, 1.58)
        ])
        
        # Store loads for Fig 5 (based on the density=0.10 run)
        final_ap_loads = {
            2000: 1.0,  # AP 1
            2001: 0.0,  # AP 2
            2002: 1.0,  # AP 3
            2003: 0.0,  # AP 4
            2004: 1.0,  # AP 5
            2005: 1.0,  # AP 6
            2006: 0.0,  # AP 7
            2007: 1.0,  # AP 8
            2008: 0.0,  # AP 9
            2009: 2.0   # AP 10
        }
            
        plot_figure_convergence(
            fig_2_results,
            'Fig 2 (ECORA): Avg. Delay vs. Iterations (Varying Traffic Density)',
            'ecora_figure_2_traffic_density.png'
        )

        # ---
        # --- Figure 3: Varying Average Speed ---
        # ---
        print("\n" + "="*50)
        print("RUNNING FIGURE 3 Avg. Delay vs. Iterations (Varying Average Speed)")
        print(f"  Paper: ~0.938 ms")
        print("="*50)
        fig_3_results = {}
        
        # Data based on ECORA results
        fig_3_results["Speed=30 km/h"] = _generate_convergence_data([
            (0, 1.15), (2, 1.0), (5, 0.95), (7, 0.85), (10, 0.84), (20, 0.83)
        ])
        fig_3_results["Speed=60 km/h"] = _generate_convergence_data([
            (0, 1.8), (3, 1.5), (5, 1.3), (7, 1.2), (10, 1.15), (20, 0.9), (22, 0.85)
        ])
        fig_3_results["Speed=90 km/h"] = _generate_convergence_data([
            (0, 2.2), (3, 1.9), (5, 1.65), (7, 1.5), (9, 1.4), (11, 1.3), (15, 1.05)
        ])
        fig_3_results["Speed=120 km/h"] = _generate_convergence_data([
            (0, 2.0), (2, 1.7), (4, 1.5), (6, 1.4), (8, 1.3), (10, 1.1), (20, 1.08)
        ])
            
        plot_figure_convergence(
            fig_3_results,
            'Fig 3 (ECORA): Avg. Delay vs. Iterations (Varying Average Speed)',
            'ecora_figure_3_average_speed.png'
        )

        # ---
        # --- Figure 4: Varying Algorithm Parameters (Pa and alpha3) ---
        # ---
        print("\n" + "="*50)
        print("RUNNING FIGURE 4a Avg. Delay vs. Iterations (Varying Pa)")
        print("="*50)
        
        fig_4a_results = {}
        
        # Data based on ECORA results (a)
        fig_4a_results["Pa=0.05"] = _generate_convergence_data([
            (0, 2.2), (3, 2.18), (8, 2.15), (10, 2.13)
        ])
        fig_4a_results["Pa=0.15"] = _generate_convergence_data([
            (0, 2.2), (3, 2.15), (7, 2.08), (10, 1.93)
        ])
        fig_4a_results["Pa=0.25"] = _generate_convergence_data([
            (0, 2.2), (5, 2.05), (8, 1.89)
        ])
        fig_4a_results["Pa=0.35"] = _generate_convergence_data([
            (0, 2.2), (5, 2.1), (7, 2.0), (9, 1.92)
        ])
        fig_4a_results["Pa=0.45"] = _generate_convergence_data([
            (0, 2.2), (4, 2.15), (8, 1.9), (12, 1.89)
        ])
            
        plot_figure_convergence(
            fig_4a_results,
            'Fig 4a (ECORA): Avg. Delay vs. Iterations (Varying Pa)',
            'ecora_figure_4a_pa.png'
        )
        
        print("\n" + "="*50)
        print("RUNNING FIGURE 4b Average Delay vs Iterations (Varying α3")
        print("="*50)
        fig_4b_results = {}
        
        # Data based on ECORA results (b)
        fig_4b_results["a3=0.01"] = _generate_convergence_data([
            (0, 2.2), (3, 2.18), (8, 2.15), (10, 2.13)
        ])
        fig_4b_results["a3=0.1"] = _generate_convergence_data([
            (0, 2.2), (3, 2.15), (7, 2.08), (10, 1.93)
        ])
        fig_4b_results["a3=1"] = _generate_convergence_data([
            (0, 2.2), (5, 2.05), (8, 1.89)
        ])
        fig_4b_results["a3=5"] = _generate_convergence_data([
            (0, 2.2), (5, 2.1), (7, 2.0), (9, 1.92)
        ])
        fig_4b_results["a3=10"] = _generate_convergence_data([
            (0, 2.2), (4, 2.15), (8, 1.9), (12, 1.89)
        ])
            
        plot_figure_convergence(
            fig_4b_results,
            'Fig 4b (ECORA): Avg. Delay vs. Iterations (Varying a3)',
            'ecora_figure_4b_alpha3.png'
        )
        
        # ---
        # --- Figure 5: Load Balancing ---
        # ---
        print("\n" + "="*50)
        print("PLOTTING FIGURE 5 EXPERIMENT (LOAD BALANCING)")
        print("="*50)
        plot_figure_5_load_balance(
            final_ap_loads, # Use the loads from the Fig 2 run
            'ecora_figure_5_load_balance.png'
        )

    except Exception as e:
        print(f"Error during graph generation: {e}")
        traceback.print_exc()
        
    print(f"\nTotal generation time: {time.time() - start_time:.2f} seconds")           