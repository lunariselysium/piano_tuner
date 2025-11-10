import numpy as np
from scipy.optimize import minimize
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

def create_stretched_reference_tuning(ref_points):
    # ... (no changes needed here, but we will call it differently)
    # Note: A 10th degree polynomial can be unstable. A 2nd or 3rd degree
    # is often more robust and produces a smoother, more realistic curve.
    # I've changed it back to 2, but feel free to experiment.
    degree = 2
    fixed_midi_notes = sorted(list(ref_points.keys()))
    if len(fixed_midi_notes) < degree + 1:
        raise ValueError(f"Need at least {degree + 1} points for a {degree}-degree polynomial fit.")
    
    x_data = np.array(fixed_midi_notes)
    y_data = []
    for midi in fixed_midi_notes:
        ideal_freq = FREQ_A4 * (2**((midi - MIDI_A4) / 12.0))
        actual_freq = ref_points[midi]
        cents_dev = 1200 * np.log2(actual_freq / ideal_freq)
        y_data.append(cents_dev)
    y_data = np.array(y_data)

    coefficients = np.polyfit(x_data, y_data, degree)
    fit_curve = np.poly1d(coefficients)
    
    stretched_ref_freqs = {}
    for midi_note in ALL_MIDI_NOTES:
        cents_dev = fit_curve(midi_note)
        ideal_freq_12tet = FREQ_A4 * (2**((midi_note - MIDI_A4) / 12.0))
        final_freq = ideal_freq_12tet * (2**(cents_dev / 1200.0))
        stretched_ref_freqs[midi_note] = final_freq
        
    return stretched_ref_freqs

def calculate_stretched_target_beat_rate(n1_midi, n2_midi, p1, p2, stretched_ref_tuning):
    # ... (no changes needed here)
    f1_ideal_stretched = stretched_ref_tuning[n1_midi]
    f2_ideal_stretched = stretched_ref_tuning[n2_midi]
    partial1_ideal_freq = p1 * f1_ideal_stretched
    partial2_ideal_freq = p2 * f2_ideal_stretched
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

# 2. CREATE A HYPOTHETICAL STRETCH CURVE FOR TARGETS
# Instead of a rigid ladder, we define a smooth target curve using just a few
# key points based on typical piano tuning principles. These points are ONLY
# used to create the target beat rates, NOT to fix the notes themselves.
print("Generating a smooth target curve...")
hypothetical_stretch_points = {
    21: FREQ_A4 * (2**((21 - MIDI_A4) / 12.0)) * (2**(-10 / 1200.0)), # A0: ~10 cents flat
    MIDI_A4: FREQ_A4,                                                # A4: 0 cents deviation
    108: FREQ_A4 * (2**((108 - MIDI_A4) / 12.0)) * (2**(+30 / 1200.0)) # C8: ~30 cents sharp
}
stretched_reference_tuning = create_stretched_reference_tuning(hypothetical_stretch_points)


# 3. GENERATE THE FULL LIST OF INTERVALS TO CHECK
# The weights inside this list now control the entire tuning hierarchy.
# Octaves have high weights, ensuring the "ladder" forms naturally.
print("Generating weighted intervals for cost function...")
intervals_to_check = intervals.generate_intervals_from_reference(
    intervals_to_generate=[
        'Octave', 'Double Octave', 'Triple Octave',
        'Perfect 5th', 'Perfect 4th',
        'Major 3rd', 'Minor 3rd',
        'Major 10th', 'Major 12th', 'Major 17th'
    ],
    reference_note_midi=MIDI_A4
)

# 4. PRE-CALCULATE ALL TARGET BEAT RATES from the smooth reference curve
print("Pre-calculating target beat rates from smooth reference...")
target_beat_rates = {}
for n1, n2, p1, p2, _ in intervals_to_check:
    key = (n1, n2, p1, p2)
    target = calculate_stretched_target_beat_rate(n1, n2, p1, p2, stretched_reference_tuning)
    target_beat_rates[key] = target

# 5. DEFINE ALL OTHER NOTES AS VARIABLES FOR THE OPTIMIZER
variable_midi_notes = sorted([m for m in ALL_MIDI_NOTES if m not in fixed_notes])

# Use our smooth stretched tuning for a much better initial guess
initial_guess = [stretched_reference_tuning[m] for m in variable_midi_notes]


# ==============================================================================
# ======================== RUN THE MAIN OPTIMIZATION ===========================
# ==============================================================================

print("\nStarting unified optimization for all notes...")
result = minimize(
    fun=main_cost_function,
    x0=initial_guess,
    args=(fixed_notes, variable_midi_notes, target_beat_rates, intervals_to_check), # Pass intervals_to_check
    method='L-BFGS-B',
    options={'eps': 1e-15, 'maxiter': 2000, 'ftol': 1e-12} # Tighter tolerance might be needed
)

if result.success:
    print("Unified optimization successful!")
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
else:
    print("Unified optimization failed:", result.message)

print("\n\n\n")
print("--- B_dict (JSON) ---")
B_dict = {midi_note: get_B_value(midi_note) for midi_note in ALL_MIDI_NOTES}
print(json.dumps(B_dict))
print("\n--- final_tuning (JSON) ---")
print(json.dumps(final_tuning))