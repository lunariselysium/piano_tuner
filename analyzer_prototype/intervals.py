from collections import deque

# Define the standard MIDI note range for an 88-key piano
PIANO_MIN_MIDI = 21  # A0
PIANO_MAX_MIDI = 108 # C8

# A dictionary containing data for common musical intervals used in tuning.
# The partials represent the harmonic ratios that are typically checked for purity.
# For example, an Octave's 2:1 ratio means the 2nd harmonic of the lower note
# should align with the 1st harmonic (fundamental) of the upper note.
DEFAULT_INTERVAL_DATA = {
    # Most fundamental intervals
    'Octave': {'semitones': 12, 'partials': (2, 1), 'weight': 4.0},
    'Perfect 5th': {'semitones': 7, 'partials': (3, 2), 'weight': 1.8},
    'Perfect 4th': {'semitones': 5, 'partials': (4, 3), 'weight': 1.8},

    # Thirds are crucial for temperament
    'Major 3rd': {'semitones': 4, 'partials': (5, 4), 'weight': 1.0},
    'Minor 3rd': {'semitones': 3, 'partials': (6, 5), 'weight': 0.8},

    # Compound intervals (larger than an octave)
    'Major 10th': {'semitones': 16, 'partials': (5, 2), 'weight': 0.7}, # Major 3rd + Octave
    'Minor 10th': {'semitones': 15, 'partials': (12, 5), 'weight': 0.6},# Minor 3rd + Octave
    'Major 12th': {'semitones': 19, 'partials': (3, 1), 'weight': 0.9}, # Perfect 5th + Octave

    # Other useful intervals
    'Major 6th': {'semitones': 9, 'partials': (5, 3), 'weight': 0.6},

    # --- New Intervals ---
    'Major 17th': {'semitones': 28, 'partials': (5, 1), 'weight': 0.65}, # Major 3rd + 2 Octaves
    'Minor 17th': {'semitones': 27, 'partials': (24, 5), 'weight': 0.55},# Minor 3rd + 2 Octaves
    'Double Octave': {'semitones': 24, 'partials': (4, 1), 'weight': 1.9}, # 2 Octaves
    'Triple Octave': {'semitones': 36, 'partials': (8, 1), 'weight': 1.6}, # 3 Octaves
}

def generate_intervals_from_reference(
    intervals_to_generate: list[str],
    reference_note_midi: int = 69, # Default to A4
    custom_weights: dict[str, float] = None
) -> list[tuple[int, int, int, int, float]]:
    """
    Generates a list of intervals by propagating outwards from a reference note.

    Args:
        intervals_to_generate: A list of interval names to propagate (e.g., ['Perfect 5th', 'Octave']).
        reference_note_midi: The MIDI note to start the propagation from. Defaults to A4 (69).
        custom_weights: A dictionary to override default weights for specific intervals.

    Returns:
        A list of unique tuples representing intervals:
        (note1_midi, note2_midi, partial1, partial2, weight).
        
    Raises:
        ValueError: If the reference note is out of bounds or an interval name is not recognized.
    """
    if not (PIANO_MIN_MIDI <= reference_note_midi <= PIANO_MAX_MIDI):
        raise ValueError(f"Reference note {reference_note_midi} is outside the piano range.")

    if custom_weights is None:
        custom_weights = {}

    # --- Initialization ---
    notes_to_process = deque([reference_note_midi])
    processed_notes = {reference_note_midi}
    generated_intervals_set = set()

    # --- Propagation Loop ---
    while notes_to_process:
        current_note = notes_to_process.popleft()

        for interval_name in intervals_to_generate:
            if interval_name not in DEFAULT_INTERVAL_DATA:
                raise ValueError(f"Unknown interval '{interval_name}'.")
            
            data = DEFAULT_INTERVAL_DATA[interval_name]
            semitones = data['semitones']
            partials = data['partials']
            weight = custom_weights.get(interval_name, data['weight'])

            # 1. Propagate UPWARDS (current_note is the lower note)
            note_up = current_note + semitones
            if note_up <= PIANO_MAX_MIDI:
                interval_tuple = (current_note, note_up, partials[0], partials[1], weight)
                generated_intervals_set.add(interval_tuple)
                if note_up not in processed_notes:
                    processed_notes.add(note_up)
                    notes_to_process.append(note_up)

            # 2. Propagate DOWNWARDS (current_note is the upper note)
            note_down = current_note - semitones
            if note_down >= PIANO_MIN_MIDI:
                # Ensure the lower note is always first in the tuple for consistency
                interval_tuple = (note_down, current_note, partials[0], partials[1], weight)
                generated_intervals_set.add(interval_tuple)
                if note_down not in processed_notes:
                    processed_notes.add(note_down)
                    notes_to_process.append(note_down)

    # Return a sorted list for predictable output
    return sorted(list(generated_intervals_set))



if __name__=="__main__":
    # --- Example Usage ---

    # 1. Define the types of intervals we want to propagate.
    #    A common temperament sequence uses Fifths, Fourths, and Octaves.
    intervals_for_temperament = [
        'Octave',
        'Perfect 5th',
        'Perfect 4th',
        'Major 3rd',
        'Minor 3rd'
    ]
    intervals_for_temperament = ['Octave','Double Octave','Triple Octave',
                                                          'Perfect 5th', 'Perfect 4th', 
                                                          'Major 3rd', 'Minor 3rd',
                                                          'Major 10th','Minor 10th',
                                                          'Major 6th','Major 12th',
                                                          'Major 17th','Minor 17th'
                                                          ]

    # # 2. (Optional) Define custom weights.
    # user_defined_weights = {
    #     'Octave': 3.0,      # Octaves must be pure, so give them the highest weight
    #     'Perfect 5th': 2.0, # The primary interval for setting the bearing
    #     'Perfect 4th': 2.0, # Equally important
    #     'Major 3rd': 1.0    # Used to check the "sweetness" of the temperament
    # }

    # 3. Generate the interval list starting from A4.
    cost_function_intervals = generate_intervals_from_reference(
        intervals_to_generate=intervals_for_temperament,
        reference_note_midi=69, # A4
        # custom_weights=user_defined_weights
    )

    # 4. Print the results for inspection.
    print(f"--- Generated Intervals by Propagation from A4 (MIDI 69) ---")
    print(f"Total unique intervals generated: {len(cost_function_intervals)}\n")

    print("Some intervals generated early in the process (centered around A4):")
    # Filter to show intervals connected to the reference note A4 (69)
    for interval in cost_function_intervals:
        if interval[0] == 69 or interval[1] == 69:
            print(interval)

    print("\nFirst 5 intervals in the final sorted list:")
    for i in range(min(5, len(cost_function_intervals))):
        print(cost_function_intervals[i])

    print("\nLast 5 intervals in the final sorted list:")
    for i in range(max(0, len(cost_function_intervals) - 5), len(cost_function_intervals)):
        print(cost_function_intervals[i])
    
    # print("\n\n\n\n\n\n YOLOOOO")
    # for i in range(0,len(cost_function_intervals)):
    #     print(cost_function_intervals[i])