import numpy as np
from skopt import gp_minimize  # Gaussian Process minimization
from skopt.space import Real, Integer  # Define search space dimensions
from skopt.plots import plot_convergence
import subprocess  # Import the subprocess module
import multiprocessing  # Import the multiprocessing module
import functools # For partial function application
import sys
import time
import os
import glob

# --- Configuration ---
JULIA_EXECUTABLE_PATH = "/home/grasinmj/julia-1.11.3/bin/julia"  # <--- **CHANGE THIS TO YOUR JULIA EXECUTABLE PATH** (e.g., /usr/bin/julia, C:\Julia\bin\julia.exe)
JULIA_SCRIPT_PATH = "AM-crystal-Potts.jl"  # <--- **CHANGE THIS TO YOUR JULIA SCRIPT PATH**
JULIA_OPTS = ["-O","3","-t","4"]
# --- End Configuration ---

# 0.a Read in simulation data
def read_crystallization(outdir):
    # 3.b Run command to post-process simulation
    command = [
        JULIA_EXECUTABLE_PATH,  # Path to the Julia interpreter
        "post-process_cg.jl",     # Path to your Julia simulation script
        outdir
    ]
    
    print(os.getpid(), "Executing command:", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    # 4. Process the Output from Julia
    output_str = result.stdout.strip()
    print(os.getpid(), "Julia Script Output:", output_str) # Print Julia's raw output

    crystallization_degree = float(output_str)  # Assuming Julia script prints the degree

    # Ensure crystallization degree is within a realistic range
    #crystallization_degree = np.clip(crystallization_degree, 0, 1)

    return crystallization_degree


# 0.b Read in previous simulation data
xinit_ = []
yinit_ = []
directory_pattern = "T0-*_Tbed-*_v0-*_trow-*_bopt"  # Your directory pattern
search_path = "data"  # Current directory as the starting search path, you can change this

# Construct the full pattern to search within the directory
full_pattern = os.path.join(search_path, directory_pattern)

# Use glob.glob() to find all paths matching the pattern
matching_paths = glob.glob(full_pattern)

# Loop through the matching paths and filter for directories
for path in matching_paths:
    if os.path.isdir(path):  # Check if it's a directory
        print(f"Found directory: {path}")
        # --- Your code to process each matching directory goes here ---
        # For example, you might want to extract parameters from the directory name:
        directory_name = os.path.basename(path) # Get just the directory name part
        parts = directory_name.split('_') # Split by underscore

        try:
            T0_value    =    float(parts[0].split('-')[1]) / 10.0       # Extract value after "T0-"
            Tbed_value  =    float(parts[1].split('-')[1]) / 10.0       # Extract value after "Tbed-"
            v0_value    =    float(parts[2].split('-')[1]) / 100.0      # Extract value after "v0-"
            trow_value  =    float(parts[3].split('-')[1]) / 10000.0    # Extract value after "trow-"

            print(f"  Extracted parameters:")
            print(f"    T0: {T0_value}, Tbed: {Tbed_value}, v0: {v0_value}, trow: {trow_value}")

            # --- Further processing within each directory ---
            # e.g., reading data files from within the directory
            # data_file_path = os.path.join(path, "simulation_output.txt")
            # if os.path.isfile(data_file_path):
            #     with open(data_file_path, 'r') as data_file:
            #         # ... process data_file ...
            
            crystallization_degree = read_crystallization(path)
            print(f"    crystallization degree: {crystallization_degree}")
            xinit_.append([T0_value, Tbed_value, v0_value, trow_value])
            yinit_.append(-crystallization_degree)

        except IndexError:
            print(f"  Warning: Could not fully parse directory name: {directory_name}")


    else:
        print(f"Warning: Path '{path}' matches pattern but is NOT a directory. Skipping.")

print("\nFinished processing matching directories.")
print("    x0 = ", xinit_)
print("    y0 = ", yinit_)

# 1. Define the Objective Function (Calling Julia Script Simulation)

def simulate_crystallization(params):
    """
    This function calls a JULIA SCRIPT (your polymer crystallization simulation)
    and reads its output to get the degree of crystallization.

    Args:
        params (list): A list of processing parameters in the order:
                       [nozzle_temp, bed_temp, velocity, thickness]

    Returns:
        float: Negative of the degree of crystallization.
               Returns a large value (e.g., 1.0) if the Julia script fails.
    """
    print(os.getpid(), f"--- simulate_crystallization called ---") # ADDED: Function call marker
    print(os.getpid(), f"Type of params: {type(params)}")      # ADDED: Check the type of params
    print(os.getpid(), f"Value of params: {params}") 

    nozzle_temp, bed_temp, velocity, thickness = params
    outdir = "data/T0-{:04d}_Tbed-{:04d}_v0-{:04d}_trow-{:04d}_bopt".format(int(10*nozzle_temp), int(10*bed_temp), int(100*velocity), int(10000*thickness))

    try:
        # 2. Construct the Command to Run the Julia Script
        command = [
            JULIA_EXECUTABLE_PATH,  # Path to the Julia interpreter
            *JULIA_OPTS,
            JULIA_SCRIPT_PATH,     # Path to your Julia simulation script
            "--T0", str(nozzle_temp),      # Parameters as command-line arguments for Julia
            "--Tbed", str(bed_temp),
            "--v0", str(velocity),
            "--trow", str(thickness),
            "--outdir", str(outdir),
            "--timeout", str(0.05),
            "--timeplot", str(0.05)
        ]
        print(os.getpid(), "Executing command:", " ".join(command))

        # 3.a Use subprocess.Popen to run the Julia script
        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,  # Capture stdout
                                   stderr=subprocess.PIPE,  # Capture stderr (optional, for error messages)
                                   text=True)              # Decode stdout and stderr as text

        # 3.a.1 Stream stdout in real-time
        print(os.getpid(), "--- Simulation Output Stream ---") # Separator for clarity
        while True:
            line = process.stdout.readline() # Read line from stdout
            if not line:
                break  # No more output, process likely finished

            print(os.getpid(), line.strip()) # Print to Python's stdout (your console), remove trailing newline
            sys.stdout.flush() # Force flush to display immediately (important for buffering)


        # 3.a.2 Wait for process to finish and get return code
        process.wait()
        return_code = process.returncode

        if return_code != 0: # Check for errors (non-zero return code)
            stderr_output = process.stderr.read() # Read any captured stderr output
            print(os.getpid(), f"Error running Julia script (Return Code: {return_code})")
            print(os.getpid(), "Stderr Output:\n", stderr_output)
            return 1.0 # Penalize failure

	    # 3.b Run command to post-process simulation
        crystallization_degree = read_crystallization(outdir)


    except subprocess.CalledProcessError as e:
        print(os.getpid(), f"Error running Julia script (Return Code: {e.returncode}):")
        print(os.getpid(), "Stdout:", e.stdout)
        print(os.getpid(), "Stderr:", e.stderr)
        return 1.0  # Penalize failure

    except FileNotFoundError:
        print(os.getpid(), f"Error: Julia executable or script not found. Check paths:")
        print(os.getpid(), f"  Julia Executable Path: {JULIA_EXECUTABLE_PATH}")
        print(os.getpid(), f"  Julia Script Path: {JULIA_SCRIPT_PATH}")
        return 1.0 # Penalize failure

    except ValueError: # If Julia output is not a valid float
        print(os.getpid(), f"Error: Could not convert Julia script output to a float. Output was: '{output_str}'")
        return 1.0 # Penalize failure

    except Exception as e: # Catch any other unexpected errors
        print(os.getpid(), f"An unexpected error occurred: {e}")
        return 1.0 # Penalize failure

    print(os.getpid(), "cost: ", type(-crystallization_degree), " ,", -crystallization_degree)
    return -crystallization_degree  # Return NEGATIVE for minimization

# 2. Define the Search Space (Parameter Bounds)
param_bounds = [
    Real(low=617.0, high=750.0, name='nozzle_temp'),     # Nozzle temperature (°K)
    Real(low=250.0, high=600.0, name='bed_temp'),        # Bed temperature (°K)
    Real(low=0.25, high=5.0, name='velocity'),           # Printer head velocity (cm/s, adjust range and units)
    Real(low=0.01, high=0.07, name='thickness')          # Row thickness (cm, adjust range and units)
]
# Explanation of Real():  skopt.space.Real defines a continuous parameter.
#  - low: Minimum allowed value.
#  - high: Maximum allowed value.
#  - name: Name of the parameter (for readability and plotting).
# You can use Integer() for integer parameters if any of your parameters are discrete.

def single_bo_run(run_id, n_calls_per_run, n_initial_points, random_seed):
    """
    Performs a single Bayesian Optimization run.

    Args:
        run_id (int): Identifier for the run (for output).
        n_calls_per_run (int): Number of function calls for this run.
        n_workers_per_run (int): Number of workers to use for parallel evaluation within this run.
        random_seed (int): Random seed for reproducibility of this run.

    Returns:
        OptimizeResult: The result object from gp_minimize.
    """
    print(f"\n--- Starting BO Run ID: {run_id} ---")

    global xinit_
    global yinit_
    print("passing {} initial points".format(len(xinit_)))

    # 3. Perform Bayesian Optimization
    result = gp_minimize(
        simulate_crystallization,
        dimensions=param_bounds,
        n_calls=n_calls_per_run,
        n_initial_points=n_initial_points, # Keep initial points
        random_state=random_seed, # Use a specific random seed for each run
        acq_func="EI",
        x0=xinit_,
        y0=yinit_
    )

    print(f"\n--- Finished BO Run ID: {run_id} ---")
    return result

# 4. Settings for Parallel Bayesian Optimization Runs

n_parallel_runs = 8  # Number of independent Bayesian Optimization runs to execute in parallel
n_calls_per_run = 25 # Number of function evaluations PER RUN. Total evaluations = n_parallel_runs * n_calls_per_run
n_calls_min = 1
n_init = len(xinit_)
n_initial_points = max(10 - n_init, 0)
n_calls = max(n_calls_per_run - n_init, n_calls_min)

# 5. Execute Parallel Bayesian Optimization Runs using multiprocessing

if __name__ == '__main__': # Recommended for multiprocessing in Python

    print(f'Found {n_init} initial data points')
    print(f'n_calls_min = {n_calls_min}')
    print(f'n_calls = {n_calls}')
    print(f'n_initial_points = {n_initial_points}')
    
    parallel_run_args = [
        (run_id, n_calls_per_run, n_initial_points, 
            (time.time_ns() + run_id) % (2**32)) # Arguments for each run
        for run_id in range(n_parallel_runs)
    ]

    with multiprocessing.Pool(processes=n_parallel_runs) as pool:
        parallel_results = pool.starmap(single_bo_run, parallel_run_args) # Run BO runs in parallel

    # 6. Find the Best Result from all Parallel Runs

    best_result = None
    best_crystallization_degree = float('-inf') # Initialize with negative infinity

    print("\n--- Summary of Results from Parallel BO Runs ---")
    for i, result in enumerate(parallel_results):
        current_crystallization_degree = -result.fun # Remember we minimized negative crystallization degree
        print(f"Run ID: {i}, Best Crystallization Degree: {current_crystallization_degree:.4f}, Parameters: {result.x}")
        if current_crystallization_degree > best_crystallization_degree:
            best_crystallization_degree = current_crystallization_degree
            best_result = result

    # 7. Analyze and Interpret Best Results

    print("Optimization Results:")
    print(f"  Best parameters: {best_result.x}")  # Best parameter values found
    print(f"  Maximum crystallization (negative objective value): {-best_result.fun:.4f}") # Max. degree of crystallization

    # Access parameter names and values more clearly:
    best_params_dict = dict(zip([d.name for d in param_bounds], best_result.x))
    print("\nBest Parameters in Dictionary Format:")
    for name, value in best_params_dict.items():
        print(f"  {name}: {value:.4f}")


    # 5. Optional: Plot Convergence (to check optimization progress)
    plot_convergence(best_result)
    import matplotlib.pyplot as plt
    plt.savefig('bopt_convergence.pdf')
    plt.show()

    # 6. Using the Best Parameters (in your actual workflow)
    best_nozzle_temp = best_params_dict['nozzle_temp']
    best_bed_temp = best_params_dict['bed_temp']
    best_velocity = best_params_dict['velocity']
    best_thickness = best_params_dict['thickness']

    print("\nTo use the best parameters in your actual simulation/printing:")
    print(f"  Nozzle Temperature: {best_nozzle_temp:.2f} °K")
    print(f"  Bed Temperature: {best_bed_temp:.2f} °K")
    print(f"  Velocity: {best_velocity:.2f} cm/s")
    print(f"  Thickness: {best_thickness:.3f} cm")

# Now you would use these 'best_' variables in your actual additive manufacturing
# process or for further, more detailed simulations around these optimal points.
