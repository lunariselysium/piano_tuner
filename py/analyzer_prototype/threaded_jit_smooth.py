import numpy as np
from scipy.optimize import minimize, basinhopping
import json
import intervals # Assuming this module is available
from numba import jit, njit
import multiprocessing
import os
import random

# --- Constants and Helper Functions ---
MIDI_A4 = 69
FREQ_A4 = 440.0
ALL_MIDI_NOTES = range(21, 109)

# --- NEW: SMOOTHING PARAMETER ---
SMOOTHNESS_WEIGHT = 40.0

@njit(cache=True)
def get_B_value(midi_note):
    """Calculates the inharmonicity coefficient B for a given MIDI note."""
    x = midi_note - 21
    B_base = 0.00005
    C_coeff = 8.06e-10
    P_exponent = 3
    B_value = B_base + C_coeff * (x**P_exponent)
    return max(0.0, B_value)

@njit(cache=True)
def get_partial_freq(fundamental, n, B):
    """Calculates the frequency of the nth partial given inharmonicity B."""
    return n * fundamental * np.sqrt(1 + B * n**2)

def create_reference_tuning():
    """Generates a dictionary for a perfect 12-TET tuning."""
    perfect_12tet_freqs = {}
    for midi_note in ALL_MIDI_NOTES:
        final_freq = FREQ_A4 * (2**((midi_note - MIDI_A4) / 12.0))
        perfect_12tet_freqs[midi_note] = final_freq
    return perfect_12tet_freqs

@njit(cache=True)
def numba_cost_function(all_frequencies, intervals_array, targets_array, b_values, smoothness_weight):
    """
    A Numba-optimized cost function that calculates the total weighted error.
    """
    total_cost = 0.0
    num_intervals = intervals_array.shape[0]

    for i in range(num_intervals):
        interval_info = intervals_array[i]
        idx1 = int(interval_info[0])
        idx2 = int(interval_info[1])
        p1 = int(interval_info[2])
        p2 = int(interval_info[3])
        weight = interval_info[4]

        f1 = all_frequencies[idx1]
        f2 = all_frequencies[idx2]
        
        B1 = b_values[idx1]
        B2 = b_values[idx2]

        actual_beat_rate = abs(get_partial_freq(f1, p1, B1) - get_partial_freq(f2, p2, B2))
        target_beat_rate = targets_array[i]
        
        beat_rate_error = (actual_beat_rate - target_beat_rate)**2
        total_cost += weight * beat_rate_error

    if smoothness_weight > 0:
        smoothness_cost = 0.0
        for i in range(1, len(all_frequencies) - 1):
            f_prev = all_frequencies[i-1]
            f_curr = all_frequencies[i]
            f_next = all_frequencies[i+1]
            
            if f_curr > 1e-6:
                jaggedness = (f_next - 2 * f_curr + f_prev) / f_curr
                smoothness_cost += jaggedness**2
        
        total_cost += smoothness_weight * smoothness_cost
        
    return total_cost

# ==============================================================================
# ======================== UNIFIED SETUP (executed once) =======================
# ==============================================================================

fixed_notes = {MIDI_A4: FREQ_A4}
reference_tuning = create_reference_tuning()

print("Generating weighted intervals...")
intervals_to_check = intervals.generate_intervals_from_reference(
    intervals_to_generate=[
        'Octave', 'Double Octave', 'Perfect 5th', 'Perfect 4th',
        'Major 3rd', 'Major 10th',
    ],
    custom_weights={
        'Octave': 4.0, 'Double Octave': 2.0, 'Perfect 5th': 3.0, 
        'Perfect 4th': 3.5, 'Major 3rd': 0.1, 'Major 10th': 0.5,
    },
    reference_note_midi=MIDI_A4
)

print("Converting data structures for Numba optimization...")
all_midi_notes_list = sorted(list(ALL_MIDI_NOTES))
midi_to_idx_map = {midi: i for i, midi in enumerate(all_midi_notes_list)}
idx_to_midi_map = {i: midi for i, midi in enumerate(all_midi_notes_list)}

intervals_array_list = []
targets_list = []
for n1, n2, p1, p2, weight in intervals_to_check:
    intervals_array_list.append([midi_to_idx_map[n1], midi_to_idx_map[n2], p1, p2, weight])
    targets_list.append(0) # Keeping original target value

intervals_array = np.array(intervals_array_list, dtype=np.float64)
targets_array = np.array(targets_list, dtype=np.float64)
b_values_array = np.array([get_B_value(midi) for midi in all_midi_notes_list], dtype=np.float64)

variable_midi_notes = sorted([m for m in ALL_MIDI_NOTES if m not in fixed_notes])
fixed_notes_keys = sorted(fixed_notes.keys())
initial_guess_base = np.array([reference_tuning[m] for m in variable_midi_notes])
variable_indices = np.array([midi_to_idx_map[m] for m in variable_midi_notes])
fixed_indices = np.array([midi_to_idx_map[m] for m in fixed_notes_keys])
fixed_freqs = np.array([fixed_notes[m] for m in fixed_notes_keys])
all_freqs_template = np.zeros(len(all_midi_notes_list))
all_freqs_template[fixed_indices] = fixed_freqs

def cost_function_wrapper(variable_freqs):
    """Wrapper for Numba cost function."""
    current_all_freqs = all_freqs_template.copy()
    current_all_freqs[variable_indices] = variable_freqs
    return numba_cost_function(current_all_freqs, intervals_array, targets_array, b_values_array, SMOOTHNESS_WEIGHT)


# ==============================================================================
# ======================== PARALLEL OPTIMIZATION LOGIC =========================
# ==============================================================================

def run_optimization_task(initial_guess):
    """
    Runs a single instance of the basinhopping optimization.
    This function is executed by a single process/core.
    """
    # For the very first task (which will be executed in the main process BEFORE pool.map),
    # we perform a quick run to ensure Numba compiles everything.
    # For subsequent tasks run by pool.map, Numba code is already compiled.
    if os.environ.get('OMP_NUM_THREADS') != '1': # Check if this is the main process run
        print("Process (main) performing initial Numba compilation/warm-up...")
        
        # Use a short, local minimization for compilation, not a full basinhopping
        # This ensures Numba sees all paths, but we don't waste time on global search.
        minimizer_kwargs_compile = {
            "method": "L-BFGS-B",
            "jac": False,
            "options": {'eps': 1e-12, 'maxiter': 10, 'ftol': 1e-12} # Very few iterations
        }
        
        # Run a quick local minimization to trigger JIT compilation
        try:
            minimize(cost_function_wrapper, initial_guess_base, **minimizer_kwargs_compile)
            print("Numba compilation/warm-up complete.")
        except Exception as e:
            print(f"Error during Numba warm-up: {e}")
            # If compilation fails here, it will likely fail later too.
            # For robustness, you might want to handle this better.


    print(f"Process {os.getpid()} starting optimization from a random point...")
    
    # Settings for the actual basinhopping runs (in parallel)
    minimizer_kwargs_basinhopping = {
        "method": "L-BFGS-B",
        "jac": False,
        "options": {'eps': 1e-12, 'maxiter': 5000, 'ftol': 1e-12}
    }

    result = basinhopping(
        func=cost_function_wrapper,
        x0=initial_guess,
        minimizer_kwargs=minimizer_kwargs_basinhopping,
        niter=300,  # Standard number of global iterations for each parallel run
        T=1.0,
        disp=False  # Keep worker output clean
    )
    
    print(f"Process {os.getpid()} finished. Final cost: {result.fun:.6f}")
    return (result.fun, result.x)


if __name__ == "__main__":
    # --- CRITICAL: CONTROL PARALLELISM ---
    num_processes = 4
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    # --- Initial Numba Warm-up Run ---
    # This single, quick run in the main process ensures all @njit functions are compiled
    # before we start spawning worker processes. This prevents each worker from recompiling.
    print("Performing initial Numba compilation/warm-up in main process...")
    # We'll use a very short local minimization for compilation
    minimizer_kwargs_compile_initial = {
        "method": "L-BFGS-B",
        "jac": False,
        "options": {'eps': 1e-12, 'maxiter': 5, 'ftol': 1e-12} # Minimal iterations for compilation
    }
    try:
        minimize(cost_function_wrapper, initial_guess_base, **minimizer_kwargs_compile_initial)
        print("Numba compilation/warm-up complete.")
    except Exception as e:
        print(f"Error during initial Numba warm-up: {e}")
        # Consider exiting or raising error if compilation is critical and fails.

    # 1. Generate a list of different random initial guesses
    print("\nGenerating randomized initial guesses for parallel runs...")
    num_variables = len(initial_guess_base)
    deviation_percent = 0.05 # 5% deviation from 12-TET
    min_bound = initial_guess_base * (1.0 - deviation_percent)
    max_bound = initial_guess_base * (1.0 + deviation_percent)

    initial_guesses = [
        np.random.uniform(low=min_bound, high=max_bound, size=num_variables)
        for _ in range(num_processes)
    ]

    # 2. Create a pool of worker processes and run the tasks
    print(f"\nStarting {num_processes} parallel optimization runs...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        # The 'run_optimization_task' function will be called for each item in 'initial_guesses'.
        # Since Numba is already compiled, the worker processes will run the full basinhopping.
        results = pool.map(run_optimization_task, initial_guesses)

    # 3. Find the best result from all the parallel runs
    best_result = min(results, key=lambda item: item[0])
    best_fun_value, best_params = best_result

    print("\n\n=============================================")
    print("ALL PARALLEL OPTIMIZATIONS FINISHED.")
    print("=============================================")
    print(f"Lowest cost found across all runs: {best_fun_value:.6f}")

    # 4. Process and display the best result
    final_tuning = dict(fixed_notes)
    for i, midi in enumerate(variable_midi_notes):
        final_tuning[midi] = best_params[i]

    print("\n--- Final Tuning Snippet (Middle C to B4) ---")
    for midi in range(60, 72):
        if midi in final_tuning:
            freq = final_tuning[midi]
            ideal_freq = FREQ_A4 * (2**((midi - MIDI_A4) / 12.0))
            cents_dev = 1200 * np.log2(freq / ideal_freq)
            print(f"MIDI {midi}: {freq:.4f} Hz (Deviation: {cents_dev:+.2f} cents)")
    
    print("\n--- final_tuning (JSON) ---")
    print(json.dumps(final_tuning, indent=2))