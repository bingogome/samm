import importlib
import os
from pathlib import Path
import sys
import traceback
import unittest

import slicer


def project_root():
    configured = os.environ.get("SAMM_TEST_ROOT")
    return Path(configured).resolve() if configured else Path(__file__).resolve().parents[1]


class SAMMModuleDiscoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = project_root()
        cls.module_path = cls.root / "samm" / "SegmentAnyMedicalModel"

    def test_module_is_discovered(self):
        self.assertTrue(hasattr(slicer.modules, "segmentanymedicalmodel"))

    def test_collision_safe_helper_package_is_loaded_from_samm(self):
        helper = importlib.import_module("samm_lib")
        helper_path = Path(helper.__file__).resolve()
        self.assertTrue(helper_path.is_relative_to(self.module_path.resolve()))


def run_tests():
    from samm_lib.test import SegmentAnyMedicalModelTest

    suite = unittest.TestSuite()
    suite.addTests(
        unittest.defaultTestLoader.loadTestsFromTestCase(SAMMModuleDiscoveryTest)
    )
    suite.addTests(
        unittest.defaultTestLoader.loadTestsFromTestCase(SegmentAnyMedicalModelTest)
    )
    return unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)


def main():
    try:
        result = run_tests()
        status = 0 if result.wasSuccessful() else 1
    except Exception:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    slicer.util.exit(status)


if __name__ == "__main__":
    main()
