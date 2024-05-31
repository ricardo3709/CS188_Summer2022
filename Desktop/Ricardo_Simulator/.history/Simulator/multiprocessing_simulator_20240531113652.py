from multiprocessing import Pool
import cProfile
from main import run_sim
import itertools

# Define variable values
REWARD_THETA = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0
]

def process_function(thetas):
    for theta in thetas:
        run_sim(theta)

def multi_process_test():
    # Determine the number of processes
    num_processes = 4
    # Split the REWARD_THETA into chunks for each process
    theta_chunks = [REWARD_THETA[i::num_processes] for i in range(num_processes)]
    
    with Pool(num_processes) as pool:
        pool.map(process_function, theta_chunks)

if __name__ == "__main__":
    multi_process_test()
