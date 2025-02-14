from multiprocessing import Pool
import cProfile
from main import run_sim
import itertools

# Define variable values
REWARD_THETA = [0.0,0.0,0.0,0.0,
                0.5,0.5,0.5,0.5,
                1.0,1.0,1.0,1.0,
                1.5,1.5,1.5,1.5,
                2.0,2.0,2.0,2.0,
                3.0,3.0,3.0,3.0,
                4.0,4.0,4.0,4.0,
                5.0,5.0,5.0,5.0,
                10.0,10.0,10.0,10.0,
                20.0,20.0,20.0,20.0]
# 'REWARD_THETA': 1.0, 'REWARD_TYPE': 'REJ', 'NODE_LAYERS': 2, 'MOVING_AVG_WINDOW': 20

# Generate all combinations of the variables
# combinations = list(itertools.product(REWARD_THETA))

# Create a list of dictionaries for each test case
# test_cases = [{'REWARD_THETA': t, 'REWARD_TYPE': r, 'NODE_LAYERS': n, 'MOVING_AVG_WINDOW': m}
#               for t, r, n, m in combinations]

def process_function(test_case):
    run_sim(test_case)  # Adjust if run_sim expects different parameters
    # Optionally use cProfile here if profiling is necessary

def multi_process_test():
    # Using a pool of workers equal to the number of cores, 4 cores
    with Pool(4) as pool:
        pool.map(process_function, REWARD_THETA)

if __name__ == "__main__":
    multi_process_test()
