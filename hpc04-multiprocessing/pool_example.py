import numpy as np
import multiprocessing
import os
slurm_procs = int(os.environ["SLURM_NTASKS"])

def sample_quarter_circle(n_samples: int):
    seed = os.getpid()
    
    rng = np.random.default_rng(seed) # use PID as seed
    x = rng.uniform(0, 1, n_samples)
    y = rng.uniform(0, 1, n_samples)
    inside_circle = (x**2 + y**2) <= 1 # return array of True and False
    n_inside_circle = np.sum(inside_circle)
    
    return n_inside_circle

num_samples_total = 1e8
sample_iterable = [int(num_samples_total / slurm_procs)] * slurm_procs

with multiprocessing.Pool(processes = slurm_procs) as p:
    n_in_circle_list = p.map(func = sample_quarter_circle, iterable = sample_iterable)

print(f"Pi estimate: {4 * np.sum(np.array(n_in_circle_list)) / num_samples_total}")