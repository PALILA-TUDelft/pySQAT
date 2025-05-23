import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from psychoacoustic_metrics import Loudness_ISO532_1

class TestLoudnessISO5321(unittest.TestCase):
    def setUp(self):
        # Create a sample input signal (sine wave)
        self.fs = 48000  # Sampling frequency
        self.duration = 10  # Duration in seconds
        self.t = np.linspace(0, self.duration, int(self.fs * self.duration), endpoint=False)
        self.insig = 0.1 * np.sin(2 * np.pi * 1000 * self.t)  # 1 kHz sine wave
        self.field = 0  # Free field
        self.method = 2  # Time-varying method
        self.time_skip = 0.1  # Skip first 0.1 seconds

    def test_valid_input(self):
        # Test with valid input
        result = Loudness_ISO532_1(self.insig, self.fs, self.field, self.method, self.time_skip, show=False)
        self.assertIsInstance(result, dict)
        self.assertIn("InstantaneousLoudness", result)
        self.assertIn("SpecificLoudness", result)
        self.assertIn("time", result)

    def test_empty_signal(self):
        # Test with an empty input signal
        with self.assertRaises(ValueError):
            Loudness_ISO532_1(np.array([]), self.fs, self.field, self.method, self.time_skip, show=False)

    def test_invalid_sampling_rate(self):
        # Test with invalid sampling rate
        with self.assertRaises(ValueError):
            Loudness_ISO532_1(self.insig, None, self.field, self.method, self.time_skip, show=False)

    def test_stationary_method(self):
        # Test with stationary method
        result = Loudness_ISO532_1(self.insig, self.fs, self.field, method=0, time_skip=self.time_skip, show=False)
        self.assertIn("Loudness", result)
        self.assertIn("SpecificLoudness", result)

    def test_show_plot(self):
        # Test with show=True (should not raise errors)
        try:
            Loudness_ISO532_1(self.insig, self.fs, self.field, self.method, self.time_skip, show=True)
        except Exception as e:
            self.fail(f"Loudness_ISO532_1 raised an exception with show=True: {e}")

    def test_white_noise(self):
        white_noise = np.random.normal(0, 0.1, len(self.t))
        result = Loudness_ISO532_1(white_noise, self.fs, self.field, self.method, self.time_skip, show=False)
        self.assertIsInstance(result, dict)
        self.assertIn("InstantaneousLoudness", result)

    def test_non_standard_sampling_rate(self):
        fs_non_standard = 44100
        result = Loudness_ISO532_1(self.insig, fs_non_standard, self.field, self.method, self.time_skip, show=False)
        self.assertIsInstance(result, dict)
        self.assertIn("InstantaneousLoudness", result)
    
    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            Loudness_ISO532_1(self.insig, self.fs, self.field, method=3, time_skip=self.time_skip, show=False)

    def test_time_skip_exceeds_duration(self):
        with self.assertRaises(ValueError):
            Loudness_ISO532_1(self.insig, self.fs, self.field, self.method, time_skip=20, show=False)

if __name__ == "__main__":
    unittest.main()