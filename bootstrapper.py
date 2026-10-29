"""Initial setup for the mae_st tooling. Import this FIRST, before anything
from `mae_st` or `custom`; importing it is the whole setup.

Two directories, kept apart on purpose:

  TOOLING_ROOT      this directory -- code only (custom/, tools/, launchers)
  EXPERIMENTS_ROOT  the user's data: configs/, work_dirs/, checkpoints/
                    default ~/mae_st_experiments, override with $MAE_ST_EXPERIMENTS

plus the two external locations:

  MAE_ST_REPO_ROOT  a clone of LCAD-UFES/mae_st_lcad, branch main (the MAE-ST
                    code with the range-image domain changes), default ~/mae_st_lcad
                    ($MAE_ST_REPO); any directory name works
  DATASETS_ROOT     default ~/datasets ($MAE_ST_DATASETS)

Side effects of importing this module, in order:
  1. TOOLING_ROOT on sys.path (so `import custom...` and `import bootstrapper`
     work from configs and from any cwd);
  2. MAE_ST_REPO_ROOT registered as the package `mae_st`, the name its code
     imports itself under.

Configs are plain Python files executed by the drivers; they only need
`import bootstrapper` and `from custom... import ...` at the top.
"""

import importlib.machinery
import importlib.util
import os
import os.path as osp
import sys

TOOLING_ROOT = osp.dirname(osp.abspath(__file__))
EXPERIMENTS_ROOT = osp.abspath(osp.expanduser(
    os.environ.get("MAE_ST_EXPERIMENTS", "~/mae_st_experiments")))
MAE_ST_REPO_ROOT = osp.abspath(osp.expanduser(
    os.environ.get("MAE_ST_REPO", "~/mae_st_lcad")))
DATASETS_ROOT = osp.abspath(osp.expanduser(
    os.environ.get("MAE_ST_DATASETS", "~/datasets")))

DATASET_PATH_PREFIX = "mae_st_overfit_tests"

CONFIGS_DIR = osp.join(EXPERIMENTS_ROOT, "configs")
WORK_DIRS = osp.join(EXPERIMENTS_ROOT, "work_dirs")
CHECKPOINTS_DIR = osp.join(EXPERIMENTS_ROOT, "checkpoints")


def _prepend(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def _register_mae_st(root):
    if not osp.isfile(osp.join(root, "models_mae.py")):
        raise FileNotFoundError(f"no mae_st clone at {root}; clone LCAD-UFES/mae_st_lcad there or set $MAE_ST_REPO")
    spec = importlib.machinery.ModuleSpec("mae_st", None, is_package=True)
    spec.submodule_search_locations = [root]
    sys.modules["mae_st"] = importlib.util.module_from_spec(spec)


_prepend(TOOLING_ROOT)
if "mae_st" not in sys.modules:
    _register_mae_st(MAE_ST_REPO_ROOT)


def work_dir_for(config_name):
    """work_dirs/<prefix>/<config name>/ under EXPERIMENTS_ROOT."""
    return osp.join(WORK_DIRS, DATASET_PATH_PREFIX, config_name)


def resolve(path, base):
    """Absolute or existing paths are returned as given (expanded); anything
    else is taken relative to `base`. This is what lets the VSCode tasks
    prompt for bare names (`my_config.py`, `kinetics400_slice10`) without
    knowing where EXPERIMENTS_ROOT or DATASETS_ROOT are."""
    path = osp.expanduser(path.strip())
    if osp.isabs(path) or osp.exists(path):
        return osp.abspath(path)
    return osp.join(base, path)


def describe():
    return (f"TOOLING_ROOT={TOOLING_ROOT}\nEXPERIMENTS_ROOT={EXPERIMENTS_ROOT}\n"
            f"MAE_ST_REPO_ROOT={MAE_ST_REPO_ROOT}\nDATASETS_ROOT={DATASETS_ROOT}")


if __name__ == "__main__":
    print(describe())
