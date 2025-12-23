import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.properties import ObjectProperty
import threading
import numpy as np

# Import our modules
import config
from hardware import AudioEngine, TunerDevice
from calculation import OptimizationService

# Fix for some Raspberry Pi window providers
import os
#os.environ['KIVY_GL_BACKEND'] = 'gl' #not used for windows testing

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

class InstructionScreen(BaseScreen):
    pass

class MeasurementScreen(BaseScreen):
    def on_enter(self):
        self.app = App.get_running_app()
        self.note_list = list(range(config.MEASURE_START_MIDI, config.MEASURE_END_MIDI + 1))
        self.current_index = 0
        self.measured_data = {}
        
        self.current_samples = [] # Buffer for 5 samples
        self.waiting_for_silence = False
        self.silence_timer = 0
        
        self.update_target_display()
        self.event = Clock.schedule_interval(self.check_audio, 0.05)

    def on_leave(self):
        Clock.unschedule(self.event)

    def update_target_display(self):
        midi = self.note_list[self.current_index]
        note_name = self.midi_to_name(midi)
        self.ids.note_target_lbl.text = note_name
        self.ids.progress_bar.value = (self.current_index / len(self.note_list)) * 100
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
            midi_target = self.note_list[self.current_index]
            
            # --- VALIDATION LOGIC ---
            ideal_freq = 440.0 * (2**((midi_target - 69) / 12.0))
            min_valid, max_valid = ideal_freq / 1.06, ideal_freq * 1.06 # +/- 1 semitone
            
            if min_valid <= detected_freq <= max_valid:
                # Accumulate sample
                self.current_samples.append(result['B'])
                count = len(self.current_samples)
                
                if count < 5:
                    self.ids.status_lbl.text = f"Sample {count}/5 Captured"
                    self.ids.status_lbl.color = config.PRIMARY
                    self.waiting_for_silence = True # Require release between samples
                    self.silence_timer = 0
                else:
                    self.ids.status_lbl.text = f"Captured: {detected_freq:.1f}Hz"
                    self.ids.status_lbl.color = config.PRIMARY
                    
                    # Calculate B average (remove lowest and highest)
                    s = sorted(self.current_samples)
                    avg_b = sum(s[1:4]) / 3
                    self.measured_data[midi_target] = avg_b
                    
                    self.current_samples = [] # Reset buffer
                    Clock.schedule_once(self.next_note, 1.0)
            else:
                msg = "Too High!" if detected_freq > ideal_freq else "Too Low!"
                self.ids.status_lbl.text = f"{msg} ({detected_freq:.0f}Hz)"
                self.ids.status_lbl.color = config.DANGER

    def next_note(self, dt=None):
        self.current_index += 1
        if self.current_index >= len(self.note_list):
            # DONE
            self.app.measured_inharmonicity = self.measured_data
            self.app.sm.current = 'calculation'
        else:
            self.waiting_for_silence = True # Require release between notes
            self.silence_timer = 0
            self.update_target_display()

    def skip_note(self):
        # Mark as 0 or estimated
        midi = self.note_list[self.current_index]
        self.measured_data[midi] = 0.0001 # Default B
        self.current_samples = []
        self.next_note()

    def midi_to_name(self, midi):
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (midi // 12) - 1
        name = notes[midi % 12]
        return f"{name}{octave}"

class CalculationScreen(BaseScreen):
    def on_enter(self):
        app = App.get_running_app()
        # Start the multiprocessing task
        app.optimizer.start_calculation(app.measured_inharmonicity)
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

        # Screen Manager
        self.sm = ScreenManager()
        self.sm.add_widget(ConnectionScreen())
        self.sm.add_widget(CalibrationScreen())
        self.sm.add_widget(InstructionScreen())
        self.sm.add_widget(MeasurementScreen())
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