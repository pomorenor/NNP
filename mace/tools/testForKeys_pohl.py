import unittest
from default_keys import DefaultKeys  # Import from your module

class TestDefaultKeys(unittest.TestCase):
    def test_keydict_contents(self):
        keys = DefaultKeys.keydict()
        self.assertEqual(keys["energy_key"], "REF_energy")
        self.assertEqual(keys["veff_key"], "Veff")
        self.assertEqual(keys["m1_key"], "M1")
        self.assertEqual(len(keys), len(DefaultKeys))  # Ensure no key is missing

if __name__ == "__main__":
    unittest.main()
