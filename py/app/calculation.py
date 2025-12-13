import time
import multiprocessing
import numpy as np

# This function would contain your heavy "script 2" logic.
# It runs in a separate PROCESS.
def heavy_calculation_task(measured_data, return_dict):
    """
    measured_data: dict of {midi_note: B_value}
    return_dict: multiprocessing manager dict to store results
    """
    print("[Calc] Process started. Optimizing curve...")
    
    # --- YOUR OPTIMIZATION LOGIC GOES HERE ---
    # For now, we simulate the calculation time and return 
    # a simple stretched tuning curve (Equal Temperament + Stretch)
    
    total_steps = 10
    for i in range(total_steps):
        time.sleep(0.5) # Simulate crunching numbers
        # Update progress (optional, would need another queue)
    
    # Mock Result: Generate a tuning map
    final_tuning = {}
    for note in range(21, 109):
        # Simple stretch logic: (note - 69) * 0.1 cents stretch
        freq_et = 440.0 * (2**((note - 69) / 12.0))
        stretch_factor = 1.0 + (0.0002 * (note-69))
        final_tuning[note] = freq_et * stretch_factor
        
    print("[Calc] Optimization complete.")
    return_dict['result'] = final_tuning
    return_dict['status'] = 'done'

class OptimizationService:
    def __init__(self):
        self.process = None
        self.manager = multiprocessing.Manager()
        self.return_dict = self.manager.dict()

    def start_calculation(self, measured_data):
        self.return_dict['status'] = 'running'
        self.return_dict['result'] = None
        
        self.process = multiprocessing.Process(
            target=heavy_calculation_task,
            args=(measured_data, self.return_dict)
        )
        self.process.start()

    def check_status(self):
        if self.process and self.process.is_alive():
            return "running"
        if self.return_dict.get('status') == 'done':
            return "done"
        return "idle"

    def get_results(self):
        return self.return_dict.get('result')