import numpy as np
from scipy.stats import zipf

def generate_zipf_popularity(num_services, alpha):
    """Generate service popularity following Zipf distribution"""
    popularity = zipf.pmf(range(1, num_services + 1), alpha)
    return popularity / popularity.sum()

def calculate_transmission_delay(data_size, distance, bandwidth):
    """Calculate transmission delay based on Shannon capacity"""
    # Simplified model
    snr = 100 / (1 + distance)  # Simple SNR model
    capacity = bandwidth * np.log2(1 + snr)  # Shannon capacity
    delay = data_size * 8 / capacity  # Convert MB to bits
    return delay

def calculate_processing_delay(computation_demand, computing_resources):
    """Calculate processing delay"""
    return computation_demand / computing_resources