import numpy as np
from scipy.optimize import minimize, basinhopping
import json
import intervals
from numba import jit, njit

# --- Constants and Helper Functions ---
MIDI_A4 = 69
FREQ_A4 = 440.0
ALL_MIDI_NOTES = range(21, 109)

# --- NEW: SMOOTHING PARAMETER ---
# Controls the penalty for "jaggedness" in the tuning curve.
# Higher values result in a smoother curve. Start with values between 1.0 and 100.0.
SMOOTHNESS_WEIGHT = 15.0

# Apply Numba's JIT compiler for significant speedup in these pure numerical functions.
# nopython=True ensures no Python interpreter overhead, and cache=True saves compilation time on subsequent runs.
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

# This is the core cost function, now heavily optimized with Numba.
# It operates exclusively on NumPy arrays for maximum performance.
@njit(cache=True)
def numba_cost_function(all_frequencies, intervals_array, targets_array, b_values, smoothness_weight):
    """
    A Numba-optimized cost function that calculates the total weighted error.
    
    Args:
        all_frequencies (np.array): Array of all frequencies for ALL_MIDI_NOTES.
        intervals_array (np.array): 2D array where each row is [idx1, idx2, p1, p2, weight].
        targets_array (np.array): 1D array of target beat rates, corresponding to intervals_array.
        b_values (np.array): Pre-calculated B values for all MIDI notes.
        smoothness_weight (float): The weight for the curve smoothing penalty.
    """
    # Part 1: Calculate the cost based on interval beat rates
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

    # Part 2: Add the smoothing penalty
    if smoothness_weight > 0:
        smoothness_cost = 0.0
        # Iterate through the interior points of the curve to check for sharp bends
        for i in range(1, len(all_frequencies) - 1):
            # Approximate the second derivative (local curvature)
            # Normalizing by f[i] makes the penalty proportional, not absolute.
            f_prev = all_frequencies[i-1]
            f_curr = all_frequencies[i]
            f_next = all_frequencies[i+1]
            
            # Avoid division by zero, though unlikely for frequencies
            if f_curr > 1e-6:
                jaggedness = (f_next - 2 * f_curr + f_prev) / f_curr
                smoothness_cost += jaggedness**2
        
        total_cost += smoothness_weight * smoothness_cost
        
    return total_cost


# ==============================================================================
# ======================== UNIFIED SETUP =======================================
# ==============================================================================

# 1. DEFINE ANCHOR NOTE AND REFERENCE TUNING
fixed_notes = {MIDI_A4: FREQ_A4}
reference_tuning = create_reference_tuning()

# 2. GENERATE INTERVALS
print("Generating weighted intervals...")
intervals_to_check = intervals.generate_intervals_from_reference(
    intervals_to_generate=[
        'Octave', 'Double Octave', 'Perfect 5th', 'Perfect 4th',
        'Major 3rd', 'Minor 3rd', 'Major 10th', 'Major 12th', 'Major 17th'
    ],
    custom_weights= {
        'Octave': 3.0, 'Perfect 5th': 2.0, 'Perfect 4th': 2.0, 'Major 3rd': 1.0,
        'Minor 3rd': 0.8, 'Major 10th': 0.6, 'Major 12th': 0.5, 'Major 17th': 0.1,
    },
    reference_note_midi=MIDI_A4
)

# 3. PREPARE DATA STRUCTURES FOR NUMBA
print("Converting data structures for Numba optimization...")

all_midi_notes_list = sorted(list(ALL_MIDI_NOTES))
midi_to_idx_map = {midi: i for i, midi in enumerate(all_midi_notes_list)}
idx_to_midi_map = {i: midi for i, midi in enumerate(all_midi_notes_list)}

intervals_array_list = []
targets_list = []
for n1, n2, p1, p2, weight in intervals_to_check:
    intervals_array_list.append([midi_to_idx_map[n1], midi_to_idx_map[n2], p1, p2, weight])
    f1_ideal = reference_tuning[n1]
    f2_ideal = reference_tuning[n2]
    target = abs((p1 * f1_ideal) - (p2 * f2_ideal))
    targets_list.append(target)

intervals_array = np.array(intervals_array_list, dtype=np.float64)
targets_array = np.array(targets_list, dtype=np.float64)
b_values_array = np.array([get_B_value(midi) for midi in all_midi_notes_list], dtype=np.float64)

# 4. SETUP OPTIMIZATION VARIABLES
variable_midi_notes = sorted([m for m in ALL_MIDI_NOTES if m not in fixed_notes])
fixed_midi_notes = sorted(fixed_notes.keys())
initial_guess = np.array([reference_tuning[m] for m in variable_midi_notes])
variable_indices = np.array([midi_to_idx_map[m] for m in variable_midi_notes])
fixed_indices = np.array([midi_to_idx_map[m] for m in fixed_midi_notes])
fixed_freqs = np.array([fixed_notes[m] for m in fixed_midi_notes])
all_freqs_template = np.zeros(len(all_midi_notes_list))
all_freqs_template[fixed_indices] = fixed_freqs

# 5. CREATE A WRAPPER FOR THE OPTIMIZER
def cost_function_wrapper(variable_freqs):
    """
    A lightweight wrapper to be called by SciPy's optimizer.
    It combines fixed and variable frequencies and calls the fast Numba function.
    """
    current_all_freqs = all_freqs_template.copy()
    current_all_freqs[variable_indices] = variable_freqs
    return numba_cost_function(current_all_freqs, intervals_array, targets_array, b_values_array, SMOOTHNESS_WEIGHT)

# ==============================================================================
# ======================== RUN THE MAIN OPTIMIZATION ===========================
# ==============================================================================

print("\nStarting optimization...")
print("Performing first run to trigger Numba compilation...")
initial_cost = cost_function_wrapper(initial_guess)
print(f"Initial cost calculated: {initial_cost:.6f}. Compilation complete.")

minimizer_kwargs = {
    "method": "L-BFGS-B",
    "jac": False,
    "options": {'eps': 1e-12, 'maxiter': 5000, 'ftol': 1e-12}
}

result = basinhopping(
    func=cost_function_wrapper,
    x0=initial_guess,
    minimizer_kwargs=minimizer_kwargs,
    niter=100,
    T=1.0,
    disp=True
)

if result.success or result.lowest_optimization_result.success:
    print("\nOptimization finished!")
    final_variable_freqs = result.x
    final_tuning = dict(fixed_notes)
    for i, midi in enumerate(variable_midi_notes):
        final_tuning[midi] = final_variable_freqs[i]

    print("\n--- Final Tuning Snippet (Middle C to B4) ---")
    for midi in range(60, 72):
        if midi in final_tuning:
            freq = final_tuning[midi]
            ideal_freq = FREQ_A4 * (2**((midi - MIDI_A4) / 12.0))
            cents_dev = 1200 * np.log2(freq / ideal_freq)
            print(f"MIDI {midi}: {freq:.4f} Hz (Deviation: {cents_dev:+.2f} cents)")
    
        print("\n\n\n")
    print("--- B_dict (JSON) ---")
    B_dict = {midi_note: get_B_value(midi_note) for midi_note in ALL_MIDI_NOTES}
    print(json.dumps(B_dict))
    print("\n--- final_tuning (JSON) ---")
    print(json.dumps(final_tuning))
    
else:
    print("\nOptimization failed or did not converge:", result.message)