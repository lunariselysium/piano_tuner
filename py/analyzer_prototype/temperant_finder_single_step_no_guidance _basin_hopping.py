#too slow

import numpy as np
from scipy.optimize import minimize, basinhopping
import json
import intervals

# --- Constants and Helper Functions (Keep these the same) ---
MIDI_A4 = 69
FREQ_A4 = 440.0
ALL_MIDI_NOTES = range(21, 109)

def get_B_value(midi_note):
    # ... (no changes needed here)
    x = midi_note - 21
    B_base = 0.00005
    C_coeff = 8.06e-10
    P_exponent = 3
    B_value = B_base + C_coeff * (x**P_exponent)
    return max(0.0, B_value)

def get_partial_freq(fundamental, n, B):
    # ... (no changes needed here)
    return n * fundamental * np.sqrt(1 + B * n**2)

def create_reference_tuning():
    """
    Generates a dictionary of frequencies for a perfect 12-Tone Equal Temperament (12-TET) tuning.

    Returns:
        dict: A dictionary where keys are MIDI note numbers (0-127) and
              values are their corresponding frequencies in Hz for a perfect 12-TET.
    """
    perfect_12tet_freqs = {}
    for midi_note in ALL_MIDI_NOTES:
        # The formula for 12-TET frequency: F = F_ref * 2^((midi_note - midi_ref) / 12)
        final_freq = FREQ_A4 * (2**((midi_note - MIDI_A4) / 12.0))
        perfect_12tet_freqs[midi_note] = final_freq
        
    return perfect_12tet_freqs

def calculate_target_beat_rate(n1_midi, n2_midi, p1, p2, ref_tuning):
    f1_ideal = ref_tuning[n1_midi]
    f2_ideal = ref_tuning[n2_midi]
    partial1_ideal_freq = p1 * f1_ideal
    partial2_ideal_freq = p2 * f2_ideal
    return abs(partial1_ideal_freq - partial2_ideal_freq)

def main_cost_function(variable_freqs, fixed_notes_dict, variable_midi_list, targets_dict, intervals_to_check):
    # ... (pass intervals_to_check as an argument)
    full_freq_dict = dict(fixed_notes_dict)
    for i, midi in enumerate(variable_midi_list):
        full_freq_dict[midi] = variable_freqs[i]

    total_cost = 0.0
    for n1, n2, p1, p2, weight in intervals_to_check:
        key = (n1, n2, p1, p2)
        if n1 in full_freq_dict and n2 in full_freq_dict and key in targets_dict:
            f1, f2 = full_freq_dict[n1], full_freq_dict[n2]
            B1, B2 = get_B_value(n1), get_B_value(n2)
            actual_beat_rate = abs(get_partial_freq(f1, p1, B1) - get_partial_freq(f2, p2, B2))
            target_beat_rate = targets_dict[key]
            beat_rate_error = (actual_beat_rate - target_beat_rate)**2
            total_cost += weight * beat_rate_error
    return total_cost

# ==============================================================================
# ======================== NEW UNIFIED SETUP ===================================
# ==============================================================================

# 1. DEFINE THE SINGLE ANCHOR NOTE
# The only truly fixed note is A4.
fixed_notes = {MIDI_A4: FREQ_A4}

# 2. SIMULATE "PERFECT" 12-TET PIANO
reference_tuning = create_reference_tuning()


# 3. GENERATE THE FULL LIST OF INTERVALS TO CHECK
# The weights inside this list now control the entire tuning hierarchy.
# Octaves have high weights, ensuring the "ladder" forms naturally.
print("Generating weighted intervals for cost function...")
intervals_to_check = intervals.generate_intervals_from_reference(
    intervals_to_generate=[
        'Octave', 'Double Octave',
        'Perfect 5th', 'Perfect 4th',
        'Major 3rd', 'Minor 3rd',
        'Major 10th', 'Major 12th', 'Major 17th'
    ],
    custom_weights= {
        'Octave': 3.0,
        'Perfect 5th': 2.0,
        'Perfect 4th': 2.0,
        'Major 3rd': 1.0,
        'Minor 3rd': 0.8,
        'Major 10th': 0.6,
        'Major 12th': 0.5,
        'Major 17th': 0.1,
    },
    reference_note_midi=MIDI_A4
)

# 4. PRE-CALCULATE ALL TARGET BEAT RATES from the reference curve
print("Pre-calculating target beat rates from reference...")
target_beat_rates = {}
for n1, n2, p1, p2, _ in intervals_to_check:
    key = (n1, n2, p1, p2)
    target = calculate_target_beat_rate(n1, n2, p1, p2, reference_tuning)
    target_beat_rates[key] = target

# 5. DEFINE ALL OTHER NOTES AS VARIABLES FOR THE OPTIMIZER
variable_midi_notes = sorted([m for m in ALL_MIDI_NOTES if m not in fixed_notes])

# Use our "perfect" 12-TET for a initial guess
initial_guess = [reference_tuning[m] for m in variable_midi_notes]


# ==============================================================================
# ======================== RUN THE MAIN OPTIMIZATION ===========================
# ==============================================================================

print("\nStarting optimization...")
# Define arguments for the local minimizer (L-BFGS-B)
minimizer_kwargs = {
    "method": "L-BFGS-B",
    "args": (fixed_notes, variable_midi_notes, target_beat_rates, intervals_to_check),
    "options": {'eps': 1e-15, 'maxiter': 500, 'ftol': 1e-13}
}
result = basinhopping(
    func=main_cost_function,
    x0=initial_guess,
    minimizer_kwargs=minimizer_kwargs,
    niter=100, # Start with 100 iterations, for example
    T=1.0
)

if result.success:
    print("Optimization successful!")
    final_tuning = dict(fixed_notes)
    for i, midi in enumerate(variable_midi_notes):
        final_tuning[midi] = result.x[i]

    print("\n--- Final Tuning Snippet (Middle C to B4) ---")
    for midi in range(60, 72):
        if midi in final_tuning:
            freq = final_tuning[midi]
            ideal_freq = FREQ_A4 * (2**((midi - MIDI_A4) / 12.0))
            cents_dev = 1200 * np.log2(freq / ideal_freq)
            print(f"MIDI {midi}: {freq:.4f} Hz (Deviation: {cents_dev:+.2f} cents)")
    
    
    # --- Calculation of MSE and R-squared ---
    print("\n--- Evaluation Metrics ---")

    actual_beat_rates_list = []
    target_beat_rates_list = []
    weights_list = []

    for n1, n2, p1, p2, weight in intervals_to_check:
        key = (n1, n2, p1, p2)
        if n1 in final_tuning and n2 in final_tuning and key in target_beat_rates:
            f1_final, f2_final = final_tuning[n1], final_tuning[n2]
            B1_final, B2_final = get_B_value(n1), get_B_value(n2)
            
            actual_beat_rate = abs(get_partial_freq(f1_final, p1, B1_final) - get_partial_freq(f2_final, p2, B2_final))
            target_beat_rate = target_beat_rates[key]

            actual_beat_rates_list.append(actual_beat_rate)
            target_beat_rates_list.append(target_beat_rate)
            weights_list.append(weight)

    actual_beat_rates_array = np.array(actual_beat_rates_list)
    target_beat_rates_array = np.array(target_beat_rates_list)
    weights_array = np.array(weights_list)
    

    # 1. Unweighted MSE on Beat Rates
    mse_unweighted = np.mean((actual_beat_rates_array - target_beat_rates_array)**2)
    print(f"Unweighted Mean Squared Error (Beat Rates): {mse_unweighted:.6f}")

    # 2. Weighted MSE on Beat Rates
    # Note: This is essentially your final cost function divided by the sum of weights,
    # or just the final cost function if you want the total weighted sum of squares.
    # To get an 'average' weighted MSE, divide by the sum of weights or number of intervals if all weights sum to N
    # For consistency with the cost function, let's keep it as sum of weighted squared errors.
    weighted_squared_errors = weights_array * ((actual_beat_rates_array - target_beat_rates_array)**2)
    mse_weighted = np.sum(weighted_squared_errors) / np.sum(weights_array) # Average weighted MSE
    print(f"Weighted Mean Squared Error (Beat Rates): {mse_weighted:.6f}")
    print(f"Total Weighted Cost (from optimization result): {result.fun:.6f}")


    # 3. Unweighted R-squared on Beat Rates
    ss_total_unweighted = np.sum((target_beat_rates_array - np.mean(target_beat_rates_array))**2)
    ss_residual_unweighted = np.sum((actual_beat_rates_array - target_beat_rates_array)**2)
    r_squared_unweighted = 1 - (ss_residual_unweighted / ss_total_unweighted) if ss_total_unweighted > 0 else 0
    print(f"Unweighted R-squared (Beat Rates): {r_squared_unweighted:.4f}")

    # 4. Weighted R-squared on Beat Rates
    # For weighted R-squared, you need to use weighted means and sums of squares.
    mean_target_weighted = np.sum(target_beat_rates_array * weights_array) / np.sum(weights_array)
    ss_total_weighted = np.sum(weights_array * (target_beat_rates_array - mean_target_weighted)**2)
    ss_residual_weighted = np.sum(weights_array * (actual_beat_rates_array - target_beat_rates_array)**2)
    r_squared_weighted = 1 - (ss_residual_weighted / ss_total_weighted) if ss_total_weighted > 0 else 0
    print(f"Weighted R-squared (Beat Rates): {r_squared_weighted:.4f}")
else:
    print("Optimization failed:", result.message)

print("\n\n\n")
# print("--- B_dict (JSON) ---")
# B_dict = {midi_note: get_B_value(midi_note) for midi_note in ALL_MIDI_NOTES}
# print(json.dumps(B_dict))
# print("\n--- final_tuning (JSON) ---")
# print(json.dumps(final_tuning))