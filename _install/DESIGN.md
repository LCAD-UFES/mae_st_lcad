# `_install/` design record — the MAE-ST environment installer

**Scope.** Getting one working Python environment for `mae_st_lcad` branch `main` (the model and data code, imported as the package `mae_st`) and this tooling, on any machine: the installer script, the dependency manifests, the lock files, and the reasoning behind every version pin. `README.md` is the runbook; this file is why it is built the way it is.

**Out of scope.** The model, the training loop and the data pipeline; their changes are documented in the commit messages of `mae_st_lcad` main and in the tooling README. Where a package problem and a code problem share a cause, this file states the package half.

**Conventions.** Every claim was verified by running a command, not inferred. Claims that could go stale carry the date they were checked. When a conclusion is overturned, it is corrected in place and dated rather than deleted.

## 1. Why this exists

MAE-ST is a loose collection of Python scripts, not an installed package, written against an old set of library versions and dependent on GPU-specific binary builds of PyTorch. In practice:

- **One dependency is pinned to a 2021 release** (`timm 0.4.5`) and cannot be bumped: newer releases moved the modules this code imports and removed an argument it passes.
- **PyTorch is not one package.** The same version number names different binaries built against different CUDA runtimes, and the wrong one imports cleanly and then cannot use the GPU.
- **On ARM machines several familiar shortcuts do not exist** — conda's `pytorch-cuda` has no `linux-aarch64` build.
- **Environments of earlier stages of this project were unusable elsewhere**: a raw `conda env export` from one x86 machine with its `prefix:` still in it, venv scripts building `detectron2` (imported nowhere) from source, and a CUDA tutorial whose first step purges the system's NVIDIA packages. They remain only in the archived history (tags `archive/main-2026-08`, `archive/installer-2026-08`).

Without automation each machine is a trial-and-error session whose result is written down nowhere. The installer makes the outcome reproducible and the reasoning inspectable: every pin sits next to the sentence explaining it.

### What the installer is

```
_install/
├── README.md              runbook and the recipe for a new manifest
├── DESIGN.md              this file
├── conda.sh               the installer; contains no package list
├── deps.<platform>.tsv    one dependency manifest per target platform
├── lock.<tag>.txt         pip list of a validated environment (gitignored)
└── logs/                  one run record per real install (gitignored)
```

`conda.sh` is bash (it needs `BASH_SOURCE` and arrays); it is not POSIX `sh`. **The manifest is a required positional argument with no default**: it is the script's input, and treating one machine's manifest as canonical is like giving a mandatory parameter a preferred value. A bare filename is resolved next to the script. Running with no argument prints usage and lists the manifests present.

## 2. Rules that are not negotiable

- **The installer never patches the repositories.** A package known to be broken is marked `required=no` so its failure is observed at verify time; editing source to route around a dependency is a code decision made by a person, in a commit.
- **Nothing fails silently.** The manifest is fully validated before anything is installed; an unknown source scheme or a missing value aborts naming the offending row.
- **Destructive actions require intent.** If the target environment exists, the user types its name to confirm deletion.
- **`--dry-run` prints every command and executes nothing**, including the run log and the lock.
- **Manifests are named after platforms, never machines or people**, like everything else in the repository.

## 3. Design

1. **The script contains no package list.** Everything installed is read from the manifest beside it.
2. **A new platform means a new manifest, never an edited script.**
3. **Every pin carries its rationale** in the manifest's `note` column. A constraint with no explanation is a future time sink: nobody can tell whether it is load-bearing or a leftover.
4. **The manifest is the intent; the lock is the result.** Pins are read back from the lock into the manifest by hand; the lock is never fed back into the installer.
5. **The lock is a step, not an artifact, and is gitignored.** Once its versions are in the manifest with their reasons, the lock is the poorer twin of the same numbers. It is reproducible by re-running the installer. The ignore rule is anchored (`_install/lock.*.txt`), so a lock deliberately moved into a subdirectory is tracked — an escape hatch that takes a decision to use.

### Manifest format

Tab-separated. `#` starts a comment, `#!` metadata the script parses (`#!target`, `#!envname`, `#!repo`). Columns: `stage · package · version · arch · installer · source · flags · required · note`.

Stages run in ascending string order, so keep two-digit zero-padded names (`00-env`, `10-build`, `20-torch`, …). Rows of one stage sharing `installer+source+flags+required` are merged into **one** command, so co-dependent packages such as torch and torchvision resolve together. The `00-env` row for `python` becomes `conda create` (its `source` supplies the channels); other `00-env` rows install normally.

**`source` grammar** — `;`-separated `scheme=value` pairs:

| installer | scheme | effect |
|---|---|---|
| pip | `pypi` | default index |
| pip | `index=<url>` | `--index-url` (replaces the default index) |
| pip | `extra=<url>` | `--extra-index-url` (searched in addition) |
| pip | `links=<url>` | `--find-links`; an HTML page or directory of `.whl` files |
| pip | `direct=<url>` | install that exact artifact, no resolution |
| pip | `git=<url>@<ref>` | `pkg @ git+<url>@<ref>`; pin a tag or a SHA, never a branch |
| pip | `local=<path>` | a local `.whl` or directory |
| conda | `channel=<a>,<b>` | `-c a -c b`, in priority order |
| conda | `nodefaults` | `--override-channels` |

`direct`, `git` and `local` override the `version` column, because the URL pins the artifact.

### Why there is no `wheel` column

A wheel's filename encodes name, version, Python, ABI and platform; there is one per *(python × ABI × platform)*, not per architecture. pip computes the host's compatible tags and picks the match, so the wheel is an **output** of resolution. What is pinned is *version + source*, and both matter: torch 2.11.0 from PyPI and `2.11.0+cu128` from `download.pytorch.org/whl/cu128` are different binaries. Changing the `python` row changes every compiled wheel in the file.

### Lessons encoded in `conda.sh`

- **conda is discovered across eight prefixes**, including `$CONDA_EXE`, miniforge and `/opt/conda`; hardcoding `~/anaconda3` either picks the wrong interpreter or wrongly reports conda missing.
- **After `conda activate`, `command -v python` must be `$CONDA_PREFIX/bin/python`**, or the script aborts: a silently failed activate installs everything into `base`.
- **torch comes from the pip index, never from conda.**
- **`set -eo pipefail`**, with `|| warn` only where deliberate (`required=no` rows).
- **GPU verification does not trust an exact string match** (§4) and finishes with a real matmul and `Conv3d` on the device.
- **Host/target divergence prints research pointers**, not a bare warning.
- **`--resolve-only` asks the resolver, not the network**: every pip stage replayed as `pip install --dry-run` for the target interpreter. sdist-only packages are `UNCHECKED`, cross-architecture runs `APPROXIMATE`, and both limits are printed.
- **Every real run writes `logs/install.<tag>.<stamp>.log` and `lock.<tag>.txt`**, the lock before verification so a failed verify still leaves the record. The lock uses `pip list --format=freeze`: `pip freeze` renders conda packages as build-machine `file://` paths and hides `pip`/`setuptools`/`wheel`.
- **(v1.3.0) A failing verification must fail the run.** `do_install` calls `verify || rc=$?`, and inside a function called in an `||` context bash ignores `set -e`: in v1.2.0 a failed import check let the function fall through to its final `ok` and return 0. `verify` now collects its own status explicitly.

## 4. GPU architecture: the check that must not be naive

`torch.cuda.get_arch_list()` lists the architectures a torch binary was **compiled for**. Testing whether the device's `sm_XY` appears in it is wrong: CUDA guarantees minor-version binary compatibility, so a cubin for `sm_XY` runs on any device of major `X` with minor ≥ `Y`. A torch whose list stops at `sm_120` runs natively on the reference machine's `sm_121`, and an exact match condemns that healthy install.

`conda.sh` parses each tag into `(major, minor)` and reports **native**, **binary-compatible**, **PTX JIT only** (works, compiles every kernel on first launch) or **no coverage**, the only real error — then runs actual GPU work, because the list is a clue and a completed matmul is evidence. The failure guarded against: torch imports, `cuda.is_available()` is True, and no kernel exists for the chip.

## 5. Reference platform and status

`deps.linux-aarch64-cu128.tsv` targets an NVIDIA DGX Spark: linux-aarch64, GB10 (Blackwell), compute capability 12.1, Ubuntu 24.04. `cuda=` in `#!target` is the newest runtime the **driver** supports, as `nvidia-smi` prints it, because that is what the preamble compares; it is not the toolkit version. The driver was 595.71.05 (CUDA 13.2) at the August validation and 580.173.02 on 2026-10-15; the manifest records the latter as `cuda=13.0`, to confirm on the next run. The cu128 wheels need a driver ≥ 570 and nothing compiles CUDA code, so both work.

### Verified 2026-08-13 (installer v1.2.0)

A real install into a scratch environment, with `CONDA_PKGS_DIRS`, `PIP_CACHE_DIR` and `TMPDIR` pointed at a scratch directory so nothing landed in shared caches (conda hardlinks from its cache on the same filesystem, so the test install is fully removable). All required packages imported, `torch 2.11.0+cu128` / `torchvision 0.26.0+cu128` installed on cp310, `sm_121` was absent from the arch list and covered by `sm_120` (§4), and the environment was 6.4 GB. The acceptance test then was the old main's `visualize_video.py` demo with the published checkpoint: it proved torch, CUDA, OpenCV and PyAV on the GPU, but not the training path.

### v1.3.0 (2026-10-28)

Moved from the `installer` branch of `mae_st_lcad` (based on code that `main` no longer has) to this tooling, which is where the environment is used. Changes: manifests named per platform; `pytorchvideo` dropped (only the old main imported it, and it cannot import beside torchvision ≥ 0.17); `wandb` and `tmux` added as optional rows, not yet validated; verification also runs the unit tests of the mae_st clone and of the tooling (the latter includes a CPU training through `tools/train.py`), checks conda rows as tools on `PATH`, and fails the run when it fails (§3); `--verify-only` re-checks an existing environment. Exercised with `--dry-run` and, against a CPU test environment, `--verify-only` (imports, both test suites, the missing-clone failure); a real install on the reference platform is pending.

## 6. Pinning policy

**Everything validated is pinned exactly**, read back from the lock: fix what works, and test migrations deliberately elsewhere. Pinning torch closes more than it seems, since torch declares its CUDA stack (`cuda-toolkit`, `cudnn`, `nccl`, `cusparselt`, `nvshmem`) with exact `==`. About twenty pure-Python transitives still float; the manifest names them under `WHAT THIS FILE DOES NOT PIN`. Their versions are not recorded once the lock is discarded — a stated limitation. The honest closure is a constraints file (`pip install -c`) supplied at install time, a mechanism not adopted; a committed lock would only look like closure while ageing separately from the manifest.

### Pins that are load-bearing

- **`timm==0.4.5` must not be bumped.** `util/video_vit.py` and `models_mae.py` import `to_2tuple` from `timm.models.layers`, and `video_vit.Attention` passes `qk_scale`; timm 0.9 moved the module and removed the argument. The wheel is `py3-none-any`.
- **`numpy` follows torch.** Torch wheels are built against a numpy ABI that torch does not declare (§7). The August pin also held a private numpy import of the old main's `video_vit.py`; current `main` has no such import, so the pin is now plain lock policy.
- **`setuptools` is held below 70** (`pkg_resources`, removed in 81, is still imported by old sdists); torch 2.11 requires `< 82`.
- **`opencv-python-headless` is a real dependency**: `util/decoder/utils.py` and `transform.py` import `cv2` at module level. The pin caught it crossing a major version (to OpenCV 5.x) on its own while left at `latest`.

### Deliberate omissions

See the manifest's `DELIBERATE OMISSIONS`: packages of earlier environments that nothing on the training path imports, scikit-learn (imported lazily by `main_test` only), a CUDA toolkit inside the env, and the repositories themselves, which are not pip packages (the tooling's bootstrapper registers the mae_st clone as `mae_st`).

## 7. Rows that depend on the torch generation

**Established 2026-08-14**, preparing a manifest for a CUDA 11.7 machine. Besides rows that follow the repository (identical everywhere) and rows that follow the machine (python, torch, its index), a third kind must agree with the **torch generation**:

- **`numpy`.** A pre-NumPy-2 torch cannot run with NumPy 2, and torch 2.0.1 does not declare numpy at all, so pip resolves the broken pair silently; no resolver, `--resolve-only` included, can check an undeclared constraint.
- **`opencv-python-headless`.** 4.12 and later require `numpy>=2`, dragging in the NumPy that breaks an older torch; 4.11.0.86 is the last accepting NumPy 1.x (cp37-abi3 wheels for x86_64 and aarch64).

The rule (README step 9): **a pin carried over from another machine's lock is unvouched**; relax it and let the local lock re-pin it. Only pins with a reason that holds everywhere (`timm==0.4.5`) travel unchanged.

## 8. Still to write

The **base + overlay split** for manifests: copying a whole manifest per platform duplicates the rows that never vary and guarantees drift. The agreed direction is a shared base plus a per-platform overlay that may override any row by package name, add rows or remove them; rows match by normalised package name, a duplicate within one file is an error, and every resolved row shows which file it came from. An overlay must be able to relax an inherited pin, not only tighten it (§7).
