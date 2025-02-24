from skopt import gp_minimize
from skopt.space import Real
import functools

# 1. Define a Minimal Objective Function
def minimal_objective_function(params):
    """
    A minimal objective function that just prints its input and returns a dummy value.
    """
    print("--- minimal_objective_function CALLED ---")
    print(f"Type of params: {type(params)}")
    print(f"Value of params: {params}")

    # If params is a list, check its length (expecting batch size if n_points > 1 works)
    if isinstance(params, list):
        print(f"Length of params list: {len(params)}")

    return 0.5  # Dummy return value


# 2. Define Search Space
param_bounds_minimal = [
    Real(low=0.0, high=1.0, name='param1'),
    Real(low=0.0, high=1.0, name='param2'),
    Real(low=0.0, high=1.0, name='param3'),
    Real(low=0.0, high=1.0, name='param4')
]

# 3. Set Optimization Parameters for Batching Test
n_calls_minimal = 5  # Keep it very small
n_initial_points_minimal = 2
n_workers_minimal = 4  # Not directly used in this minimal example, but for context
n_points_minimal = n_workers_minimal # Test with n_points = n_workers


# 4. Perform Bayesian Optimization with Minimal Example
result_minimal = gp_minimize(
    func=minimal_objective_function, # Use the minimal objective function
    dimensions=param_bounds_minimal,
    n_calls=n_calls_minimal,
    n_initial_points=n_initial_points_minimal,
    random_state=42,
    n_points=n_points_minimal, # Set n_points for batching
    acq_func="EI"
)

print("\n--- Minimal Example Optimization Finished ---")
print("Best result:", result_minimal.fun)
print("Best parameters:", result_minimal.x)
