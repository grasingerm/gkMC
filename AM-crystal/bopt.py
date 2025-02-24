import numpy as np
from skopt import gp_minimize  # Gaussian Process minimization
from skopt.space import Real, Integer  # Define search space dimensions
from skopt.plots import plot_convergence
import subprocess  # Import the subprocess module
import multiprocessing  # Import the multiprocessing module
import functools # For partial function application
import sys

# 1. Define the Objective Function (Calling Julia Script Simulation)

# --- Configuration ---
JULIA_EXECUTABLE_PATH = "/home/grasinmj/julia-1.10.4/bin/julia"  # <--- **CHANGE THIS TO YOUR JULIA EXECUTABLE PATH** (e.g., /usr/bin/julia, C:\Julia\bin\julia.exe)
JULIA_SCRIPT_PATH = "AM-crystal-Potts.jl"  # <--- **CHANGE THIS TO YOUR JULIA SCRIPT PATH**
JULIA_OPTS = ["-O","3","-t","4"]
# --- End Configuration ---

# 2. Parallel Objective Function Wrapper

def parallel_simulate_crystallization(parameter_sets, n_workers=4):
    """
    Parallelizes the execution of simulate_crystallization for multiple parameter sets
    using multiprocessing and pool.map (corrected version).

    Args:
        parameter_sets (list of lists): A list where each element is a list of processing parameters
                                       (e.g., [[params1], [params2], [params3], ...])
        n_workers (int): Number of parallel worker processes.

    Returns:
        list: A list of negative crystallization degrees.
    """
    
    print(f"--- parallel_simulate_crystallization called ---") # ADDED: Function call marker
    print(f"Type of parameter_sets: {type(parameter_sets)}")   # ADDED: Check type of parameter_sets
    print(f"Value of parameter_sets: {parameter_sets}")         # ADDED: Print value of parameter_sets

    # Ensure parameter_sets is ALWAYS treated as a list of lists ---
    if not isinstance(parameter_sets[0], list): # Check if the first element is NOT a list
        parameter_sets = [parameter_sets] # If not, wrap it in a list to make it a list of lists
    # --- Now parameter_sets is guaranteed to be a list of lists ---

    with multiprocessing.Pool(processes=n_workers) as pool:
        # Use pool.map to apply simulate_crystallization to each parameter set in parallel
        results = pool.map(simulate_crystallization, parameter_sets) # Directly pass parameter_sets

    return results

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
    print(f"--- simulate_crystallization called ---") # ADDED: Function call marker
    print(f"Type of params: {type(params)}")      # ADDED: Check the type of params
    print(f"Value of params: {params}") 

    nozzle_temp, bed_temp, velocity, thickness = params
    outdir = "data/T0-{}_Tbed-{}_v0-{}_trow-{}".format(nozzle_temp, bed_temp, velocity, thickness)

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
            "--outdir", str(outdir)
        ]
        print("Executing command:", " ".join(command))

        # 3.a Use subprocess.Popen to run the Julia script
        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,  # Capture stdout
                                   stderr=subprocess.PIPE,  # Capture stderr (optional, for error messages)
                                   text=True)              # Decode stdout and stderr as text

        # 3.a.1 Stream stdout in real-time
        print("--- Simulation Output Stream ---") # Separator for clarity
        while True:
            line = process.stdout.readline() # Read line from stdout
            if not line:
                break  # No more output, process likely finished

            print(line.strip()) # Print to Python's stdout (your console), remove trailing newline
            sys.stdout.flush() # Force flush to display immediately (important for buffering)


        # 3.a.2 Wait for process to finish and get return code
        process.wait()
        return_code = process.returncode

        if return_code != 0: # Check for errors (non-zero return code)
            stderr_output = process.stderr.read() # Read any captured stderr output
            print(f"Error running Julia script (Return Code: {return_code})")
            print("Stderr Output:\n", stderr_output)
            return 1.0 # Penalize failure

	# 3.b Run command to post-process simulation
        command2 = [
            JULIA_EXECUTABLE_PATH,  # Path to the Julia interpreter
            "post-process_cg.jl",     # Path to your Julia simulation script
            outdir
        ]
        
        print("Executing command:", " ".join(command2))
        result = subprocess.run(command2, capture_output=True, text=True, check=True)

        # 4. Process the Output from Julia
        output_str = result.stdout.strip()
        print("Julia Script Output:", output_str) # Print Julia's raw output

        crystallization_degree = float(output_str)  # Assuming Julia script prints the degree

        # Ensure crystallization degree is within a realistic range
        crystallization_degree = np.clip(crystallization_degree, 0, 1)


    except subprocess.CalledProcessError as e:
        print(f"Error running Julia script (Return Code: {e.returncode}):")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        return 1.0  # Penalize failure

    except FileNotFoundError:
        print(f"Error: Julia executable or script not found. Check paths:")
        print(f"  Julia Executable Path: {JULIA_EXECUTABLE_PATH}")
        print(f"  Julia Script Path: {JULIA_SCRIPT_PATH}")
        return 1.0 # Penalize failure

    except ValueError: # If Julia output is not a valid float
        print(f"Error: Could not convert Julia script output to a float. Output was: '{output_str}'")
        return 1.0 # Penalize failure

    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        return 1.0 # Penalize failure


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


# 3. Perform Bayesian Optimization

n_calls = 50  # Number of optimization iterations
n_initial_points = 10
n_workers = 4   # Number of parallel processes

result = gp_minimize(
    func=functools.partial(parallel_simulate_crystallization, n_workers=n_workers), # Use the parallel objective function
    dimensions=param_bounds,
    n_calls=n_calls,
    n_initial_points=n_initial_points,
    random_state=42,
    n_points=n_workers, # Number of points to evaluate in each batch
    acq_func="EI", # Expected Improvement acquisition function
)

# 4. Analyze and Interpret Results

print("Optimization Results:")
print(f"  Best parameters: {result.x}")  # Best parameter values found
print(f"  Maximum crystallization (negative objective value): {-result.fun:.4f}") # Max. degree of crystallization

# Access parameter names and values more clearly:
best_params_dict = dict(zip([d.name for d in param_bounds], result.x))
print("\nBest Parameters in Dictionary Format:")
for name, value in best_params_dict.items():
    print(f"  {name}: {value:.4f}")


# 5. Optional: Plot Convergence (to check optimization progress)
plot_convergence(result)
import matplotlib.pyplot as plt
plt.show()


# 6. Using the Best Parameters (in your actual workflow)

best_nozzle_temp = best_params_dict['nozzle_temp']
best_bed_temp = best_params_dict['bed_temp']
best_velocity = best_params_dict['velocity']
best_thickness = best_params_dict['thickness']

print("\nTo use the best parameters in your actual simulation/printing:")
print(f"  Nozzle Temperature: {best_nozzle_temp:.2f} °C")
print(f"  Bed Temperature: {best_bed_temp:.2f} °C")
print(f"  Velocity: {best_velocity:.2f} mm/s")
print(f"  Thickness: {best_thickness:.3f} mm")

# Now you would use these 'best_' variables in your actual additive manufacturing
# process or for further, more detailed simulations around these optimal points.
