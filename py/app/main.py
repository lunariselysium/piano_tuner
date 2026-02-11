import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.properties import NumericProperty
import threading
import numpy as np

# Import our modules
import config
from hardware import AudioEngine, TunerDevice
from calculation import OptimizationService
import intervals

# Fix for some Raspberry Pi window providers
import os
#os.environ['KIVY_GL_BACKEND'] = 'gl' #not used for windows testing


class StringVisualizer(Widget):
    active_index = NumericProperty(1) # 0=Left, 1=Center, 2=Right


class BaseScreen(Screen):
    pass

class ConnectionScreen(BaseScreen):
    def on_enter(self):
        # Trigger scan when screen loads
        self.ids.status_lbl.text = "Scanning for devices..."
        self.ids.bt_spinner.values = ["Scanning..."]
        
        app = App.get_running_app()
        # Call scan on background thread
        app.tuner.scan_devices(self.update_device_list)

    def update_device_list(self, devices):
        # This runs on background thread, schedule UI update on main thread
        Clock.schedule_once(lambda dt: self._update_spinner_ui(devices), 0)

    def _update_spinner_ui(self, devices):
        if not devices:
            self.ids.bt_spinner.values = ["No Devices Found"]
            self.ids.status_lbl.text = "Scan failed or no devices."
            return

        # Create "Name (Address)" strings for the spinner
        self.device_map = {f"{name} ({addr})": addr for name, addr in devices}
        self.ids.bt_spinner.values = list(self.device_map.keys())
        
        # Select the first one (which is our target if found, due to sorting)
        if self.ids.bt_spinner.values:
            self.ids.bt_spinner.text = self.ids.bt_spinner.values[0]
            self.ids.status_lbl.text = "Select Device & Connect"

    def connect_devices(self):
        selection = self.ids.bt_spinner.text
        
        if selection not in getattr(self, 'device_map', {}):
            self.ids.status_lbl.text = "Invalid Device Selection"
            return
            
        address = self.device_map[selection]
        mic_name = self.ids.mic_spinner.text
        
        # 1. Init Audio
        try:
            app = App.get_running_app()
            app.audio.start_stream(device_id=None) 
            
            self.ids.status_lbl.text = "Connecting Bluetooth..."
            
            # 2. Init BT (Async Callback)
            def on_connect_result(success):
                Clock.schedule_once(lambda dt: self._on_connect_ui(success), 0)
            
            app.tuner.connect(address, on_connect_result)
            
        except Exception as e:
            self.ids.status_lbl.text = f"Error: {str(e)}"

    def _on_connect_ui(self, success):
        app = App.get_running_app()
        if success:
            self.ids.status_lbl.text = "Connected!"
            self.ids.status_lbl.color = (0,1,0,1)
            Clock.schedule_once(lambda dt: setattr(app.sm, 'current', 'calibration'), 1)
        else:
            self.ids.status_lbl.text = "Connection Failed"
            self.ids.status_lbl.color = (1,0,0,1)

class CalibrationScreen(BaseScreen):
    def on_enter(self):
        self.event = Clock.schedule_interval(self.update_ui, 0.05)
    
    def on_leave(self):
        Clock.unschedule(self.event)

    def update_ui(self, dt):
        app = App.get_running_app()
        rms = app.audio.rms
        threshold = self.ids.sense_slider.value
        
        # Numeric feedback for easier calibration
        self.ids.numeric_lbl.text = f"RMS: {int(rms):,} | Threshold: {int(threshold):,}"
        
        # Normalize RMS visually (Reference 10M for 32-bit audio interfaces)
        visual_max = 10000000
        norm_val = min(100, (rms / visual_max) * 100) 
        self.ids.rms_bar.value = norm_val
        
        # Update Threshold Line Position on the bar
        bar = self.ids.rms_bar
        threshold_pct = min(1.0, threshold / visual_max)
        line_x = bar.x + (bar.width * threshold_pct)
        
        self.ids.threshold_line.pos = (line_x, bar.y)
        self.ids.threshold_line.height = bar.height

    def update_sensitivity(self, value):
        app = App.get_running_app()
        app.audio.sensitivity = value

class MeasurementConfigScreen(BaseScreen):
    def save_config(self):
        app = App.get_running_app()
        
        # 1. Save Sample Count
        app.sample_count_target = int(self.ids.sample_slider.value)
        
        # 2. Generate Note List
        start = config.MEASURE_START_MIDI
        end = config.MEASURE_END_MIDI
        pattern = self.ids.pattern_spinner.text
        
        if pattern == 'Every Note (Standard)':
            app.target_midi_list = list(range(start, end + 1))
        elif pattern == 'Every Other Note':
            app.target_midi_list = list(range(start, end + 1, 2))
        elif pattern == 'Octaves Only':
            app.target_midi_list = list(range(start, end + 1, 12))
        elif pattern == 'Temperament Only (F3-F4)':
            app.target_midi_list = list(range(53, 66)) # F3 to F4
            
        print(f"Config Saved: {len(app.target_midi_list)} notes, {app.sample_count_target} samples each.")

class InstructionScreen(BaseScreen):
    pass

class MeasurementScreen(BaseScreen):
    current_string_idx = NumericProperty(1) # Used by KV

    def on_enter(self):
        self.app = App.get_running_app()
        self.note_list = self.app.target_midi_list if self.app.target_midi_list else [69]

        # Structure: measured_data[midi] = { 'L': val, 'C': val, 'R': val }
        self.measured_data = {}
        
        self.note_index = 0
        self.current_samples = []
        self.waiting_for_silence = False
        self.silence_timer = 0
        
        # Initialize the queue for the first note
        self.setup_string_queue(self.note_list[self.note_index])

        self.event = Clock.schedule_interval(self.check_audio, 0.05)

    def on_leave(self):
        Clock.unschedule(self.event)

    def get_strings_for_midi(self, midi):
        """
        Returns list of string indices for a given note.
        Standard Piano Breakpoints (Approximate):
        21-28 (A0-E1): 1 String (Index 1 or 0, let's use 1 as 'Center' visually for singles)
        29-43 (F1-G2): 2 Strings (Index 0, 1 -> Left, Right? Or Left, Center?)
                       Let's use 0 (Left) and 2 (Right) for bichords to be distinct.
        44-108: 3 Strings (0, 1, 2)
        """
        if midi < 29: return [1] # Monochord (treat as Center)
        if midi < 44: return [0, 2] # Bichord (Left, Right)
        return [0, 1, 2] # Trichord

    def setup_string_queue(self, midi):
        """Prepare the list of strings to measure for the current note."""
        self.string_queue = self.get_strings_for_midi(midi)
        self.string_queue_index = 0
        self.update_display()

    def update_display(self):
        midi = self.note_list[self.note_index]
        current_str_code = self.string_queue[self.string_queue_index]
        
        # Update Visuals
        self.ids.note_target_lbl.text = self.midi_to_name(midi)
        self.current_string_idx = current_str_code
        
        # Instruction Text
        if current_str_code == 1 and len(self.string_queue) == 1:
            txt = "Play Key (Single String)"
        elif current_str_code == 0:
            txt = "Mute Center/Right -> Play LEFT"
        elif current_str_code == 1:
            txt = "Mute Left/Right -> Play CENTER"
        elif current_str_code == 2:
            txt = "Mute Left/Center -> Play RIGHT"
        self.ids.instruction_lbl.text = txt
        
        # Progress
        total_steps = len(self.note_list)
        self.ids.progress_bar.value = (self.note_index / total_steps) * 100
        self.ids.status_lbl.text = "Listening..."
        self.ids.status_lbl.color = (1,1,1,1)

    def check_audio(self, dt):
        # Silence check to prevent "speeding through" notes or getting stuck
        if self.waiting_for_silence:
            # Adjusted Threshold: Use 100% of sensitivity instead of 50%.
            # This prevents getting stuck if the noise floor is close to sensitivity.
            silence_thresh = self.app.audio.sensitivity * 1
            
            if self.app.audio.rms < silence_thresh:
                self.silence_timer += dt
                if self.silence_timer > 0.5: # 0.5s of silence required
                    self.waiting_for_silence = False
                    # Visual update to show we are ready again
                    self.ids.status_lbl.text = "Listening..."
                    self.ids.status_lbl.color = (1,1,1,1)
            else:
                self.silence_timer = 0
                # DEBUG INFO: Show current RMS vs Threshold so user knows why it's waiting
                self.ids.status_lbl.text = f"Release Key ({int(self.app.audio.rms)} > {int(silence_thresh)})"
                self.ids.status_lbl.color = config.ACCENT_1
            
            # CRITICAL: Discard stray analysis results while waiting for silence
            if self.app.audio.ready_for_analysis:
                self.app.audio.ready_for_analysis = False
            
            return

        # Poll audio engine for results
        if self.app.audio.ready_for_analysis:
            result = self.app.audio.last_analysis_result
            self.app.audio.ready_for_analysis = False # Reset flag
            
            detected_freq = result['freq']
            midi_target = self.note_list[self.note_index]
            
            # --- VALIDATION LOGIC ---
            ideal_freq = 440.0 * (2**((midi_target - 69) / 12.0))
            min_valid, max_valid = ideal_freq / 1.06, ideal_freq * 1.06 # +/- 1 semitone
            
            if min_valid <= detected_freq <= max_valid:
                self.current_samples.append(result['B'])
                count = len(self.current_samples)
                
                if count < self.app.sample_count_target:
                    self.ids.status_lbl.text = f"Sample {count}/{self.app.sample_count_target}"
                    self.waiting_for_silence = True 
                else:
                    # DONE WITH THIS STRING
                    avg_b = sum(self.current_samples) / len(self.current_samples)
                    
                    # Save Data: note -> string_idx -> value
                    midi = self.note_list[self.note_index]
                    str_idx = self.string_queue[self.string_queue_index]
                    
                    if midi not in self.measured_data:
                        self.measured_data[midi] = {}
                    
                    self.measured_data[midi][str_idx] = avg_b
                    
                    self.current_samples = []
                    
                    # Move to next string or next note
                    Clock.schedule_once(self.advance_step, 0.5)
            else:
                msg = "Too High!" if detected_freq > ideal_freq else "Too Low!"
                self.ids.status_lbl.text = f"{msg} ({detected_freq:.0f}Hz)"
                self.ids.status_lbl.color = config.DANGER

    def advance_step(self, dt):
        self.string_queue_index += 1
        
        if self.string_queue_index >= len(self.string_queue):
            # Done with all strings for this note
            self.note_index += 1
            if self.note_index >= len(self.note_list):
                # Done with all notes
                self.app.measured_inharmonicity = self.measured_data
                self.app.sm.current = 'tuning_preset'
            else:
                self.setup_string_queue(self.note_list[self.note_index])
                self.waiting_for_silence = True
        else:
            # Next string, same note
            self.update_display()
            self.waiting_for_silence = True

    def skip_note(self):
        # Skip current string
        midi = self.note_list[self.note_index]
        str_idx = self.string_queue[self.string_queue_index]
        
        if midi not in self.measured_data:
            self.measured_data[midi] = {}
        
        # Store None to indicate skip/estimate
        self.measured_data[midi][str_idx] = None 
        self.current_samples = []
        self.advance_step(None)

    def midi_to_name(self, midi):
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (midi // 12) - 1
        name = notes[midi % 12]
        return f"{name}{octave}"
    
class TuningPresetScreen(BaseScreen):
    oct_val = NumericProperty(3.0)
    fifth_val = NumericProperty(2.0)
    third_val = NumericProperty(0.8)

    def on_enter(self):
        self.ids.preset_spinner.values = list(intervals.TUNING_PRESETS.keys()) + ['Custom']
        self.ids.preset_spinner.text = 'Classic (Balanced)'
    
    def on_preset_change(self, selection):
        if selection in intervals.TUNING_PRESETS:
            preset = intervals.TUNING_PRESETS[selection]
            # Update the properties; the sliders will react automatically
            self.oct_val = preset.get('Octave', 3.0)
            self.fifth_val = preset.get('Perfect 5th', 2.0)
            self.third_val = preset.get('Major 3rd', 0.8)

    def apply_preset(self):
        app = App.get_running_app()
        selection = self.ids.preset_spinner.text
        
        if selection == 'Custom':
            # Use slider values for core weights
            app.tuning_weights = {
                'Octave': self.oct_val,
                'Perfect 5th': self.fifth_val,
                'Major 3rd': self.third_val,
                'Double Octave': 2.0, 
                'Perfect 4th': 1.5
            }
        else:
            app.tuning_weights = intervals.TUNING_PRESETS[selection]
        
        app.sm.current = 'calculation'

class CalculationScreen(BaseScreen):
    def on_enter(self):
        app = App.get_running_app()
        # Start the multiprocessing task
        app.optimizer.start_calculation(app.measured_inharmonicity, app.tuning_weights)
        self.event = Clock.schedule_interval(self.check_process, 0.5)

    def check_process(self, dt):
        app = App.get_running_app()
        status = app.optimizer.check_status()
        
        if status == 'done':
            app.tuning_targets = app.optimizer.get_results()
            Clock.unschedule(self.event)
            app.sm.current = 'tuning'


class TuningScreen(BaseScreen):
    def on_enter(self):
        self.app = App.get_running_app()
        # Create ordered list of targets
        self.target_keys = sorted(self.app.tuning_targets.keys())
        self.current_idx = 0
        self.update_screen()
        self.event = Clock.schedule_interval(self.tuning_loop, 0.1)

    def on_leave(self):
        Clock.unschedule(self.event)

    def update_screen(self):
        midi = self.target_keys[self.current_idx]
        target_f = self.app.tuning_targets[midi]
        
        self.ids.target_note_lbl.text = self.get_note_name(midi)
        self.ids.target_freq_lbl.text = f"{target_f:.2f} Hz"
        self.ids.current_freq_lbl.text = "-- Hz"
        self.ids.needle.pos_hint = {'center_x': 0.5, 'center_y': 0.45}

    def tuning_loop(self, dt):
        if self.app.audio.ready_for_analysis:
            res = self.app.audio.last_analysis_result
            self.app.audio.ready_for_analysis = False
            
            curr_freq = res['freq']
            midi = self.target_keys[self.current_idx]
            target_freq = self.app.tuning_targets[midi]
            
            # Noise filter: Ignore sounds far from target
            if abs(curr_freq - target_freq) > 100:
                return

            # Cents deviation formula
            if curr_freq > 0 and target_freq > 0:
                cents = 1200 * np.log2(curr_freq / target_freq)
            else:
                cents = 0

            self.ids.current_freq_lbl.text = f"{curr_freq:.1f} Hz"

            # Update Needle (Visual mapping: +/- 50 cents range)
            # Center is 0.5. Range is 0.0 to 1.0
            pos_calc = 0.5 + (cents / 100.0)
            pos_calc = max(0.1, min(0.9, pos_calc)) # Clamp
            pos = float(pos_calc) # Numpy to standard float
            self.ids.needle.pos_hint = {'center_x': pos, 'center_y': 0.45}
            
            # Tuning Logic
            if abs(cents) < 2: # Tolerance
                self.ids.action_log.text = "IN TUNE!"
                self.ids.action_log.color = (0,1,0,1)
            else:
                self.ids.action_log.color = config.PRIMARY
                if cents < 0:
                    self.ids.action_log.text = "Sending: UP"
                    self.app.tuner.send_command("STEP_UP", abs(cents))
                else:
                    self.ids.action_log.text = "Sending: DOWN"
                    self.app.tuner.send_command("STEP_DOWN", abs(cents))

    def next_note(self):
        if self.current_idx < len(self.target_keys) - 1:
            self.current_idx += 1
            self.update_screen()

    def get_note_name(self, midi):
        # Same helper
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (midi // 12) - 1
        return f"{notes[midi % 12]}{octave}"

class PianoTunerApp(App):
    colors = config # Make config accessible in KV
    
    def build(self):
        self.title = "Pi Piano Tuner"

        Builder.load_file('tuner.kv') 
        
        # Init Logic Modules
        self.audio = AudioEngine()
        self.tuner = TunerDevice()
        self.optimizer = OptimizationService()
        
        # State Data
        self.measured_inharmonicity = {}
        self.tuning_targets = {}

        # Measurement Config Data
        self.sample_count_target = 5
        self.target_midi_list = []

        self.tuning_weights = intervals.TUNING_PRESETS['Classic (Balanced)'] # Default

        # Screen Manager
        self.sm = ScreenManager()
        self.sm.add_widget(ConnectionScreen())
        self.sm.add_widget(CalibrationScreen())
        self.sm.add_widget(MeasurementConfigScreen())
        self.sm.add_widget(InstructionScreen())
        self.sm.add_widget(MeasurementScreen())
        self.sm.add_widget(TuningPresetScreen())
        self.sm.add_widget(CalculationScreen())
        self.sm.add_widget(TuningScreen())
        
        return self.sm

    def get_audio_devices(self):
        # Helper for Spinner
        try:
            devs = self.audio.get_devices()
            return [d['name'] for d in devs]
        except:
            return ["Default Mic"]
        
    def on_stop(self):
        self.audio.stop_stream()

if __name__ == "__main__":
    PianoTunerApp().run()