import unittest
import numpy as np
from psychoacoustic_metrics import Tonality_Aures1985

class TestTonalityAures1985(unittest.TestCase):
    def setUp(self):
        # Create a sample input signal (sine wave)
        self.fs = 44100  # Sampling frequency
        self.duration = 100  # Duration in seconds
        self.t = np.linspace(0, self.duration, int(self.fs * self.duration), endpoint=False)
        self.insig = 0.1 * np.sin(2 * np.pi * 1000 * self.t)  # 1 kHz sine wave
        self.loudness_field = 0  # Free field
        self.time_skip = 0.1  # Skip first 0.1 seconds

    def test_valid_input(self):
        # Test with valid input
        result = Tonality_Aures1985(self.insig, self.fs, self.loudness_field, self.time_skip, show=False)
        self.assertIsInstance(result, dict)
        self.assertIn("InstantaneousTonality", result)
        self.assertIn("TonalWeighting", result)
        self.assertIn("LoudnessWeighting", result)
        self.assertIn("time", result)

    def test_empty_signal(self):
        # Test with an empty input signal
        with self.assertRaises(ValueError):
            Tonality_Aures1985(np.array([]), self.fs, self.loudness_field, self.time_skip, show=False)

    def test_unsupported_sampling_rate(self):
        # Test with unsupported sampling rate
        unsupported_fs = 32000
        result = Tonality_Aures1985(self.insig, unsupported_fs, self.loudness_field, self.time_skip, show=False)
        self.assertEqual(result["time"].shape[0], len(self.insig) // (44100 // unsupported_fs))

    def test_time_skip_handling(self):
        # Test correct handling of time_skip
        result = Tonality_Aures1985(self.insig, self.fs, self.loudness_field, self.time_skip, show=False)
        time_vector = result["time"]
        self.assertGreaterEqual(time_vector[0], self.time_skip)

    def test_show_plot(self):
        # Test with show=True (should not raise errors)
        try:
            Tonality_Aures1985(self.insig, self.fs, self.loudness_field, self.time_skip, show=True)
        except Exception as e:
            self.fail(f"Tonality_Aures1985 raised an exception with show=True: {e}")

if __name__ == "__main__":
    unittest.main()