import numpy as np
from scipy.optimize import minimize
import json


import intervals

# --- Constants ---
MIDI_A4 = 69
FREQ_A4 = 440.0
ALL_MIDI_NOTES = range(21, 109) # Standard 88-key piano (A0 to C8)

def get_B_value(midi_note):
    """
    Magic function by Gemini
    Provides a more realistic, general-purpose inharmonicity coefficient B
    based on MIDI note, using a power law model.
    
    This function aims to approximate typical piano inharmonicity:
    - Starts very low in the bass.
    - Rises slowly in the mid-range.
    - Increases more rapidly in the high treble.
    
    *** IMPORTANT: This should ultimately be replaced
    with a curve fitted to actual measurements from specific piano. ***
    """
    x = midi_note - 21  # Normalize MIDI to 0-87 for 88 keys (A0 to C8)

    # These coefficients are derived from targeting plausible B values
    # at A0, A4, and C8. The power of 3.5 provides a good curve shape.
    
    B_base = 0.00005  # Base inharmonicity for the lowest notes (e.g., A0)
    
    # Coefficient and exponent for the power law growth.
    # Chosen to make B rise smoothly and reach realistic treble values.
    C_coeff = 8.06e-10 
    P_exponent = 3
    
    B_value = B_base + C_coeff * (x**P_exponent)
    
    # Ensure B doesn't go negative (though with these params, it shouldn't)
    return max(0.0, B_value)

def get_partial_freq(fundamental, n, B):
    """Calculates the frequency of the nth inharmonic partial."""
    return n * fundamental * np.sqrt(1 + B * n**2)


def create_stretched_reference_tuning(fixed_notes):
    """
    Creates a full 88-note reference tuning by fitting a smooth polynomial curve
    (in cents) to the known, stretched 'A' notes. This avoids the "sagging"
    of linear interpolation.
    
    Args:
        fixed_notes (dict): A dictionary of {midi: frequency} for the 'A' ladder.
        
    Returns:
        dict: A dictionary of {midi: frequency} for all 88 piano notes,
              representing a smooth, stretched reference tuning.
    """
    fixed_midi_notes = sorted(list(fixed_notes.keys()))

    if len(fixed_midi_notes) < 3:
        raise ValueError("Need at least three fixed 'A' notes for a stable polynomial fit.")

    # 1. Get the x (midi) and y (cents deviation) data for our fixed points
    x_data = np.array(fixed_midi_notes)
    y_data = []
    for midi in fixed_midi_notes:
        ideal_freq = FREQ_A4 * (2**((midi - MIDI_A4) / 12.0))
        actual_freq = fixed_notes[midi]
        cents_dev = 1200 * np.log2(actual_freq / ideal_freq)
        y_data.append(cents_dev)
    y_data = np.array(y_data)

    # 2. Fit a polynomial to the data. A 2nd-degree (quadratic) is a great start.
    #    A 3rd-degree (cubic) can be used for more complex curves.
    degree = 10
    coefficients = np.polyfit(x_data, y_data, degree)
    fit_curve = np.poly1d(coefficients)

    # 3. Use the fitted curve to calculate the ideal cent deviation for ALL notes
    stretched_ref_freqs = {}
    for midi_note in ALL_MIDI_NOTES:
        # Get the cent deviation from our smooth polynomial curve
        cents_dev = fit_curve(midi_note)
        
        # Convert this ideal cent deviation back into a frequency
        ideal_freq_12tet = FREQ_A4 * (2**((midi_note - MIDI_A4) / 12.0))
        final_freq = ideal_freq_12tet * (2**(cents_dev / 1200.0))
        stretched_ref_freqs[midi_note] = final_freq
        
    return stretched_ref_freqs

def calculate_stretched_target_beat_rate(n1_midi, n2_midi, p1, p2, stretched_ref_tuning):
    """
    Calculates the target beat rate for a PERFECTLY HARMONIC piano that has been
    tuned to the STRETCHED reference curve.
    """
    # Look up the "ideal" frequencies from our new stretched reference
    f1_ideal_stretched = stretched_ref_tuning[n1_midi]
    f2_ideal_stretched = stretched_ref_tuning[n2_midi]
    
    # Calculate harmonic partials from these stretched fundamentals
    partial1_ideal_freq = p1 * f1_ideal_stretched
    partial2_ideal_freq = p2 * f2_ideal_stretched
    
    return abs(partial1_ideal_freq - partial2_ideal_freq)




### Cost Function
def main_cost_function(variable_freqs, fixed_notes_dict, variable_midi_list, targets_dict):
    """
    Calculates cost as the squared difference between actual beat rates
    and the ideal 12-TET target beat rates.
    """
    full_freq_dict = dict(fixed_notes_dict)
    for i, midi in enumerate(variable_midi_list):
        full_freq_dict[midi] = variable_freqs[i]

    total_cost = 0.0
    
    for n1, n2, p1, p2, weight in intervals_to_check:
        key = (n1, n2, p1, p2)
        if n1 in full_freq_dict and n2 in full_freq_dict and key in targets_dict:
            f1, f2 = full_freq_dict[n1], full_freq_dict[n2]
            B1, B2 = get_B_value(n1), get_B_value(n2)
            
            # Calculate the ACTUAL beat rate with inharmonicity
            actual_beat_rate = abs(get_partial_freq(f1, p1, B1) - get_partial_freq(f2, p2, B2))
            
            # Get the pre-calculated TARGET beat rate
            target_beat_rate = targets_dict[key]
            
            # The error is the squared difference from the target
            beat_rate_error = (actual_beat_rate - target_beat_rate)**2
            total_cost += weight * beat_rate_error
            
    return total_cost










### Generate Octave Ladder

fixed_notes = {MIDI_A4: FREQ_A4}

# Calculate 'A's above A4
for i in range(MIDI_A4, 108, 12):
    ref_midi = i
    target_midi = i + 12
    if target_midi > 108: break
    
    ref_freq = fixed_notes[ref_midi]
    ref_B = get_B_value(ref_midi)
    target_B = get_B_value(target_midi)

    # Objective function to find the best stretched octave
    def octave_cost(target_freq):
        # Minimize the 4:2 beat rate
        p1_freq = get_partial_freq(ref_freq, 4, ref_B)
        p2_freq = get_partial_freq(target_freq, 2, target_B)
        return (p1_freq - p2_freq)**2

    # Initial guess is a pure 2:1 octave
    initial_guess = ref_freq * 2.0
    res = minimize(octave_cost, initial_guess, method='Nelder-Mead')
    fixed_notes[target_midi] = res.x[0]

# Calculate 'A's below A4
for i in range(MIDI_A4, 21, -12):
    ref_midi = i
    target_midi = i - 12
    if target_midi < 21: break
    
    ref_freq = fixed_notes[ref_midi]
    ref_B = get_B_value(ref_midi)
    target_B = get_B_value(target_midi)

    def octave_cost(target_freq):
        p1_freq = get_partial_freq(target_freq, 4, target_B)
        p2_freq = get_partial_freq(ref_freq, 2, ref_B)
        return (p1_freq - p2_freq)**2

    initial_guess = ref_freq / 2.0
    res = minimize(octave_cost, initial_guess, method='Nelder-Mead')
    fixed_notes[target_midi] = res.x[0]

print("Fixed 'A' Note Frequencies (Stretched):")
for midi, freq in sorted(fixed_notes.items()):
    print(f"MIDI {midi}: {freq:.4f} Hz")




# Define a comprehensive set of intervals across the keyboard
intervals_to_check = intervals.generate_intervals_from_reference(['Octave',
                                                        #   'Double Octave','Triple Octave',
                                                          'Perfect 5th', 'Perfect 4th', 
                                                          'Major 3rd', 'Minor 3rd',
                                                        #   'Major 10th','Minor 10th', #sends things off
                                                          'Major 6th','Major 12th',
                                                        #   'Major 17th','Minor 17th'
                                                          ])

# Calculate target beat rates from a "perfect" 12-TFT piano fitted to octave ladder
stretched_reference_tuning = create_stretched_reference_tuning(fixed_notes)
print(stretched_reference_tuning)

print("Pre-calculating target beat rates from STRETCHED reference...")
target_beat_rates = {}
for n1, n2, p1, p2, _ in intervals_to_check:
    key = (n1, n2, p1, p2)
    target = calculate_stretched_target_beat_rate(n1, n2, p1, p2, stretched_reference_tuning)
    target_beat_rates[key] = target



# Identify which notes are variables for the optimizer
variable_midi_notes = sorted([m for m in ALL_MIDI_NOTES if m not in fixed_notes])

# Create an initial guess for the variable notes based on 12-TET
initial_guess = [FREQ_A4 * (2**((m - MIDI_A4) / 12.0)) for m in variable_midi_notes]



print("\nStarting main optimization to match 12-TET beat rates...")
result = minimize(
    fun=main_cost_function,
    x0=initial_guess,
    # Pass the targets dictionary as an argument
    args=(fixed_notes, variable_midi_notes, target_beat_rates),
    method='L-BFGS-B',
    options={'eps': 1e-10, 'maxiter': 2000}
)

if result.success:
    print("Main optimization successful!")
    # Combine fixed notes and optimized results into the final tuning
    final_tuning = dict(fixed_notes)
    for i, midi in enumerate(variable_midi_notes):
        final_tuning[midi] = result.x[i]
    
    # Print a snippet of the result
    print("\n--- Final Tuning Snippet (Middle C to B4) ---")
    for midi in range(60, 72):
        freq = final_tuning[midi]
        cents_dev = 1200 * np.log2(freq / (FREQ_A4 * (2**((midi - MIDI_A4) / 12.0))))
        print(f"MIDI {midi}: {freq:.4f} Hz (Deviation: {cents_dev:+.2f} cents)")
else:
    print("Main optimization failed:", result.message)

print("\n\n\n")
print("--- B_dict (JSON) ---")
B_dict = {midi_note: get_B_value(midi_note) for midi_note in ALL_MIDI_NOTES}
print(json.dumps(B_dict))
print("\n--- final_tuning (JSON) ---")
print(json.dumps(final_tuning))