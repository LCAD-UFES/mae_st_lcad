import importlib.machinery, importlib.util, pathlib, sys

# The code imports itself as `mae_st.*`; register the repository root under that name so the tests run from any clone directory.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if "mae_st" not in sys.modules:
    spec = importlib.machinery.ModuleSpec("mae_st", None, is_package=True)
    spec.submodule_search_locations = [str(ROOT)]
    sys.modules["mae_st"] = importlib.util.module_from_spec(spec)
