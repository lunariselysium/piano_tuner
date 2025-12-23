import numpy as np
from scipy.optimize import minimize, basinhopping
import multiprocessing
import os
import json
from numba import njit
import time

# Custom module
import intervals

# --- CONSTANTS ---
MIDI_A4 = 69
FREQ_A4 = 440.0
# Full piano range (A0 to C8)
ALL_MIDI_NOTES = range(21, 109) 

# --- NUMBA JIT FUNCTIONS (Must be at top level for multiprocessing pickling) ---

@njit(cache=True)
def get_theoretical_B(midi_note):
    """Fallback formula if real data is missing."""
    x = midi_note - 21
    B_base = 0.00005
    C_coeff = 8.06e-10
    P_exponent = 3
    B_value = B_base + C_coeff * (x**P_exponent)
    return max(0.0, B_value)

@njit(cache=True)
def get_partial_freq(fundamental, n, B):
    """Calculates frequency of nth partial with inharmonicity B."""
    return n * fundamental * np.sqrt(1 + B * n**2)

@njit(cache=True)
def numba_cost_function(variable_freqs, full_freqs_template, variable_indices, 
                       intervals_array, targets_array, b_values):
    """
    Numba-optimized cost function.
    Reconstructs the full frequency array and calculates interval beat beats.
    """
    # 1. Reconstruct full array
    current_freqs = full_freqs_template.copy()
    # Manual loop for Numba compatibility (faster than array slicing sometimes)
    for i in range(len(variable_indices)):
        idx = variable_indices[i]
        current_freqs[idx] = variable_freqs[i]

    total_cost = 0.0
    num_intervals = intervals_array.shape[0]

    for i in range(num_intervals):
        # Unpack interval data
        idx1 = int(intervals_array[i, 0])
        idx2 = int(intervals_array[i, 1])
        p1   = int(intervals_array[i, 2])
        p2   = int(intervals_array[i, 3])
        weight = intervals_array[i, 4]

        f1 = current_freqs[idx1]
        f2 = current_freqs[idx2]
        B1 = b_values[idx1]
        B2 = b_values[idx2]

        # Calculate Coincident Partial Frequencies
        part1 = get_partial_freq(f1, p1, B1)
        part2 = get_partial_freq(f2, p2, B2)
        
        # Beat rate difference
        actual_beat_rate = abs(part1 - part2)
        target_beat_rate = targets_array[i]
        
        # Squared Error
        total_cost += weight * ((actual_beat_rate - target_beat_rate)**2)
        
    return total_cost

# --- HELPER: Worker Function for Parallel Processing ---
def worker_optimization_task(args):
    """
    This runs inside a separate process.
    args: tuple containing all necessary data arrays and configuration
    """
    (initial_guess, full_template, var_indices, 
     intervals_arr, targets_arr, b_vals, seed) = args
    
    # Set random seed for this worker to ensure diversity
    np.random.seed(seed)
    
    # Add slight random noise to initial guess to explore different local minima
    perturbation = np.random.uniform(-0.5, 0.5, size=initial_guess.shape)
    worker_guess = initial_guess + perturbation

    # Wrapper for scipy
    def func_wrapper(x):
        return numba_cost_function(x, full_template, var_indices, 
                                   intervals_arr, targets_arr, b_vals)

    # Local minimizer settings
    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "jac": False,
        "options": {'eps': 1e-8, 'maxiter': 1000, 'ftol': 1e-9}
    }

    # Run Basinhopping (Global Optimization)
    # Reduced niter for demo speed, increase for precision
    result = basinhopping(
        func=func_wrapper,
        x0=worker_guess,
        minimizer_kwargs=minimizer_kwargs,
        niter=30, 
        T=1.0,
        disp=False
    )
    
    return result.fun, result.x

# --- MAIN PROCESS LOGIC ---

def heavy_calculation_task(measured_data, return_dict):
    """
    Main entry point for the calculation process.
    Prepares data, spawns worker pool, aggregates results.
    measured_data structure: { MIDI_INT: { STRING_IDX: B_FLOAT, ... }, ... }
    """
    try:
        print("[Calc] Process started.")
        
        # 1. GENERATE REFERENCE (12-TET)
        all_midi_notes_list = sorted(list(ALL_MIDI_NOTES))
        reference_tuning = {}
        for midi in all_midi_notes_list:
            reference_tuning[midi] = FREQ_A4 * (2**((midi - MIDI_A4) / 12.0))

        # 2. PREPARE INTERVALS
        # Using the intervals logic provided
        intervals_list = intervals.generate_intervals_from_reference(
            intervals_to_generate=[
                'Octave', 'Double Octave', 'Perfect 5th', 'Perfect 4th',
                'Major 3rd', 'Major 10th', 'Major 12th', 'Major 17th'
            ],
            reference_note_midi=MIDI_A4,
            custom_weights={
                'Octave': 3.0, 'Double Octave': 2.0, 'Perfect 5th': 2.0, 
                'Perfect 4th': 1.5, 'Major 3rd': 0.8, 'Major 10th': 0.6,
                'Major 17th': 0.4
            }
        )
        
        # 3. CONVERT TO NUMPY ARRAYS
        midi_to_idx = {m: i for i, m in enumerate(all_midi_notes_list)}
        
        intervals_data = []
        targets_data = []
        
        for n1, n2, p1, p2, weight in intervals_list:
            # Only include if notes are within our range
            if n1 in midi_to_idx and n2 in midi_to_idx:
                intervals_data.append([midi_to_idx[n1], midi_to_idx[n2], p1, p2, weight])
                targets_data.append(0.0) # Target beat rate is 0 (pure intervals)

        intervals_arr = np.array(intervals_data, dtype=np.float64)
        targets_arr = np.array(targets_data, dtype=np.float64)

        # 4. PREPARE B VALUES (Measured + Fallback)
        b_values_list = []
        for midi in all_midi_notes_list:
            
            final_b = None
            
            if midi in measured_data and measured_data[midi]:
                strings_dict = measured_data[midi]
                valid_b_samples = []
                
                # Iterate through strings (0, 1, 2)
                for s_idx, b_val in strings_dict.items():
                    if b_val is not None and b_val > 0:
                        valid_b_samples.append(b_val)
                
                if valid_b_samples:
                    # AGGREGATION LOGIC:
                    # Unisons must be tuned to the same fundamental.
                    # We average the inharmonicity coefficient (B) of the strings
                    # to optimize for the "composite" sound of the key.
                    final_b = np.mean(valid_b_samples)

            if final_b is not None:
                b_values_list.append(final_b)
            else:
                # Fallback to theory
                b_values_list.append(get_theoretical_B(midi))
        
        b_vals_arr = np.array(b_values_list, dtype=np.float64)

        # 5. SETUP VARIABLES
        fixed_notes = {MIDI_A4: FREQ_A4}
        variable_midi = sorted([m for m in all_midi_notes_list if m not in fixed_notes])
        
        # Template for full array
        full_freqs_template = np.zeros(len(all_midi_notes_list))
        for m, f in fixed_notes.items():
            full_freqs_template[midi_to_idx[m]] = f
            
        var_indices = np.array([midi_to_idx[m] for m in variable_midi], dtype=np.int64)
        initial_guess = np.array([reference_tuning[m] for m in variable_midi], dtype=np.float64)

        # 6. PARALLEL EXECUTION
        # Determine CPU count. Reserve 1 for UI.
        total_cores = multiprocessing.cpu_count()
        workers_count = max(1, total_cores - 1)
        # Limit to 3 max to prevent overheating on Pi if it has 4 cores
        workers_count = min(3, workers_count) 
        
        print(f"[Calc] Starting pool with {workers_count} workers...")
        
        # Prepare arguments for each worker
        worker_args = []
        for i in range(workers_count):
            seed = int(time.time()) + i
            worker_args.append((
                initial_guess, 
                full_freqs_template, 
                var_indices, 
                intervals_arr, 
                targets_arr, 
                b_vals_arr, 
                seed
            ))

        with multiprocessing.Pool(processes=workers_count) as pool:
            results = pool.map(worker_optimization_task, worker_args)

        # 7. AGGREGATE RESULTS
        # Find result with lowest cost
        best_result = min(results, key=lambda x: x[0])
        best_cost, best_params = best_result
        
        print(f"[Calc] Optimization done. Best Cost: {best_cost:.4f}")

        # 8. CONSTRUCT FINAL DICT
        final_tuning = dict(fixed_notes)
        for i, midi in enumerate(variable_midi):
            final_tuning[midi] = best_params[i]

        return_dict['result'] = final_tuning
        return_dict['status'] = 'done'

    except Exception as e:
        print(f"[Calc] Error: {e}")
        import traceback
        traceback.print_exc()
        return_dict['status'] = 'error'

class OptimizationService:
    def __init__(self):
        self.process = None
        self.manager = multiprocessing.Manager()
        self.return_dict = self.manager.dict()

    def start_calculation(self, measured_data):
        """
        Starts the heavy calculation in a separate process.
        measured_data: dict {midi_number: B_value}
        """
        # Clean previous state
        self.return_dict['status'] = 'running'
        self.return_dict['result'] = None
        
        if self.process and self.process.is_alive():
            self.process.terminate()

        self.process = multiprocessing.Process(
            target=heavy_calculation_task,
            args=(measured_data, self.return_dict)
        )
        self.process.start()

    def check_status(self):
        if self.process and self.process.is_alive():
            return "running"
        
        status = self.return_dict.get('status', 'idle')
        if status == 'done' and self.process:
            self.process.join() # Clean up
            self.process = None
            
        return status

    def get_results(self):
        return self.return_dict.get('result')