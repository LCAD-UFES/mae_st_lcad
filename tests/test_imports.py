import importlib, unittest
from tests import _package  # noqa: F401

MODULES = ["mae_st.util.misc", "mae_st.util.meters", "mae_st.models_mae", "mae_st.util.kinetics"]

class ImportTest(unittest.TestCase):
    def test_modules_import_without_meta_internal_or_optional_packages(self):
        for name in MODULES:
            with self.subTest(module=name): importlib.import_module(name)

if __name__ == "__main__": unittest.main()
