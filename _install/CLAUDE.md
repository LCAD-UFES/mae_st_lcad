# CLAUDE.md — `_install/`, the MAE-ST environment installer

**Scope.** This file covers one thing: getting a working Python environment for this repository, on any machine. It owns the installer script, the dependency manifests, the lock files, and the reasoning behind every version pin.

**Out of scope, deliberately.** Nothing here describes the model, the training loop, or the known defects in `main_pretrain.py`. Those live in the experiment notes kept outside this repository. Where a package problem and a code problem share a cause, this file states the *package* half and says so; it does not restate the code half.

**Conventions.** Every claim here was verified by running a command, not inferred. Claims that could go stale carry the date they were checked. When a conclusion is overturned, it is corrected in place and dated rather than deleted, so the next reader does not re-derive it.

## 1. Why this exists

### What the library is

**MAE-ST** ("MAE as Spatiotemporal Learners") is a self-supervised pretraining method for video. A masked autoencoder cuts its input into fixed patches, hides most of them, feeds only the visible ones to a transformer encoder, and asks a small decoder to reconstruct what was hidden; the loss is computed only on the hidden parts. MAE-ST extends that from images to video by using 3D tubelets — several frames × 16 × 16 pixels — instead of flat 2D patches. Nothing is labelled: the supervision comes from the data hiding part of itself.

For installation purposes what matters is narrower: this is a **loose collection of Python scripts**, not an installed package, written against a specific and now-old set of library versions, and depending on GPU-specific binary builds of PyTorch.

### Why installing it is hard

Every one of these was hit in practice, not imagined:

- **The bundled `environment.yml` does not work and is incomplete.** It is a raw `conda env export` from somebody else's machine — it still carries their `prefix:` — and it omits five packages the training path actually imports.
- **One dependency is pinned to a version from 2021** and cannot be bumped: newer releases moved the modules this code imports and deleted an argument it passes.
- **One dependency is permanently broken** and installs anyway, on purpose, so that the failure is observed rather than silently worked around.
- **PyTorch is not one package.** The same version number names different binaries built against different CUDA runtimes, and picking the wrong one produces an install that imports cleanly and then cannot use the GPU.
- **On ARM machines, several familiar shortcuts do not exist at all** — conda's `pytorch-cuda` package, for instance, has no `linux-aarch64` build.
- **The repository also ships a CUDA installation tutorial that is actively dangerous** on a shared machine; see §2.

Without automation this becomes a long trial-and-error session per machine, and worse, the *result* of that session is not written down anywhere. The installer exists to make the outcome reproducible and, more importantly, to make the reasoning inspectable: every pin sits next to the sentence explaining why it is there.

### What the installer is

Two files that travel together, plus the artifacts a run produces:

```
_install/
├── CLAUDE.md              this file
├── conda.sh               the installer; contains no package list
├── deps.<host>.tsv        one dependency manifest per target machine
├── lock.<host>.txt        pip freeze of a validated environment (gitignored)
└── logs/                  one run record per real install (gitignored)
```

`conda.sh` is bash and needs `BASH_SOURCE` and arrays; it is not POSIX `sh`.

Usage:

```
./conda.sh <manifest.tsv>             # required positional argument; no default
./conda.sh <manifest.tsv> --env NAME  # override the manifest's #!envname
./conda.sh <manifest.tsv> --yes       # skip the menu, install directly
./conda.sh <manifest.tsv> --dry-run   # print every command, run nothing
./conda.sh                            # no manifest: usage error, lists what exists
```

**The manifest is a required positional argument and has no default.** An earlier version defaulted to one particular file, which was wrong in principle: the manifest is the script's input, and treating one machine's manifest as canonical is like giving a function a preferred value for a mandatory parameter. A bare filename is resolved next to the script; anything containing a slash is used as given. Running with no argument prints usage and lists the manifests present, so the error teaches instead of scolding.

## 2. Rules that are not negotiable

- **The installer never patches the repository.** A package that is known to be broken is marked `required=no` in the manifest so the failure is *observed* at verify time. Editing source to route around a dependency problem is a code decision made by a person, never an install step.
- **`CUDA_128_INSTALLATION_TUTORIAL.md` in this repository must be ignored.** It targets Ubuntu 20.04, desktop RTX cards and `_amd64.deb` packages, and its first section runs `apt --purge remove "nvidia*" "cuda*"`. On a shared machine with a vendor-provisioned CUDA, that is destructive to everyone using it.
- **`create_venv_cu117.sh` and `create_venv_cu128.sh` are superseded by this installer.** They build `detectron2` from source, which nothing in the repository imports, and they `mkdir checkpoints results` *inside* the clone.
- **Nothing fails silently.** The manifest is fully validated before anything is installed; an unknown source scheme or a missing value aborts naming the offending row.
- **Destructive actions require intent.** If the target environment already exists, the user must type its name to confirm deletion.
- **`--dry-run` prints every command and executes nothing**, including the run log and the lock file.

## 3. Design

1. **The script contains no package list.** Everything installed is read from the manifest beside it, located via `${BASH_SOURCE[0]}`.
2. **A new machine means a new manifest, never an edited script.** Copy whichever existing manifest is nearest, edit the copy, and pass it as the positional argument. `README.md` has the step-by-step recipe for deciding what each row should say.
3. **Every pin carries its rationale in the manifest**, in the `note` column, next to the pin. A version constraint with no explanation is a future time sink: nobody later can tell whether it is load-bearing or a leftover.
4. **The manifest is the intent; the lock is the result.** The `.tsv` records what was asked for and why. `lock.<host>.txt` is a `pip freeze` of what the resolver actually produced, transitive dependencies included. Pins are read back from the lock into the manifest **by hand**; the lock is never fed back into the installer.
5. **The lock is a step, not an artifact, and is gitignored.** It exists to be read once, during the pinning of a manifest. After that the manifest contains its versions *and* the rationale for each, so the lock is strictly the poorer of the two documents while looking equally authoritative — the classic setup for a stale twin. It is also reproducible: re-running the installer against a pinned manifest regenerates it. The ignore rule is anchored to this directory (`_install/lock.*.txt`), so a lock deliberately moved into a subdirectory is tracked normally; that escape hatch is intentional and undocumented in the runbook precisely so that using it takes a decision.

### Manifest format

Tab-separated. `#` starts a comment, `#!` starts metadata the script parses (`#!target`, `#!envname`, `#!repo`).

Columns: `stage · package · version · arch · installer · source · flags · required · note`.

Stages run in ascending string order, so keep the two-digit zero-padded convention (`00-env`, `10-build`, `20-torch`, …) — otherwise `100-x` sorts before `20-x`. Rows in the same stage that share `installer+source+flags+required` are merged into **one** command, so co-dependent packages such as torch and torchvision are resolved together rather than sequentially.

The `00-env` row for `python` is special: it becomes `conda create`, so its `installer` must be `conda`, and its `source` supplies the channels. Every other `00-env` row is installed normally.

**`source` grammar** — a `;`-separated list of `scheme=value` pairs:

| installer | scheme | effect |
|---|---|---|
| pip | `pypi` | default index |
| pip | `index=<url>` | `--index-url` (replaces the default index) |
| pip | `extra=<url>` | `--extra-index-url` (searched in addition) |
| pip | `links=<url>` | `--find-links`; an HTML page or directory of `.whl` files |
| pip | `direct=<url>` | install that exact artifact, no resolution |
| pip | `git=<url>@<ref>` | `pkg @ git+<url>@<ref>`; pin a tag or a SHA, never a branch |
| pip | `local=<path>` | a local `.whl` or directory |
| conda | `channel=<a>,<b>` | `-c a -c b`, written in priority order |
| conda | `nodefaults` | `--override-channels` |

`direct`, `git` and `local` override the `version` column, because the URL already pins the artifact.

### Why there is no `wheel` column

A wheel is a prebuilt package whose filename encodes what it is built for: `torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl` is name, version, Python version, ABI, platform. There is one wheel per *(python × ABI × platform)* combination, **not** one per architecture — torch 2.7.1 ships 24 files on PyPI for exactly that reason. pip computes the host's compatible tag set and picks the match, so the wheel is an **output** of resolution, not an input; a `wheel` column would record a result as though it were a choice.

What you actually pin is *version + source*, and both halves matter: torch 2.7.1 from PyPI and torch 2.7.1 from `download.pytorch.org/whl/cu128` are different binaries built against different CUDA runtimes. To pin one exact artifact anyway, use `direct=`. And note that **changing the `python` row changes every compiled wheel in the file.**

### Lessons encoded in `conda.sh`

- **conda is discovered across eight prefixes**, including `$CONDA_EXE`, miniforge and `/opt/conda`. Machines differ; hardcoding `~/anaconda3` means either using the wrong interpreter or wrongly reporting that conda is missing.
- **After `conda activate`, the script compares `command -v python` against `$CONDA_PREFIX/bin/python` and aborts on mismatch.** A silently failed activate installs the entire dependency set into `base`, which looks like success right up until it breaks something else on the machine.
- **torch comes from the pip index, never from conda.** `pytorch-cuda` has no `linux-aarch64` build at all.
- **`set -eo pipefail`**, with `|| warn` only where deliberate (`required=no` rows).
- **GPU verification does not trust an exact string match** — see §4 — and finishes with a real matmul and a real `Conv3d` on the device. The functional test is the verdict.
- **Host/target divergence prints research pointers, not a bare warning**: the `download.pytorch.org/whl/` index listing, the PyPI JSON query for checking platform wheels, and `conda search --platform`.
- **`--resolve-only` asks the resolver, not the network.** It replays every pip stage as `pip install --dry-run` for the manifest's target interpreter, so a nonexistent version or an unsatisfiable pair is found in seconds rather than gigabytes into an install. Two deliberate limits are stated in its own output rather than hidden: sdist-only packages are `UNCHECKED` because they cannot be resolved without building, and cross-architecture runs are `APPROXIMATE` because `--platform` needs a hand-written tag list that can never be complete — an early version of that list omitted `manylinux_2_27` and condemned a manifest known to install correctly. When the architectures match, the flag is dropped and pip's own tag set decides.
- **Every real run writes two artifacts**: `logs/install.<tag>.<stamp>.log` (commands in order, with outcomes, the manifest's sha256, and `conda list`) and `lock.<tag>.txt`. The lock is written *before* verification, so a failed verify still leaves the record needed to diagnose it.
- **The lock uses `pip list --format=freeze`, not `pip freeze`.** `pip freeze` renders conda-installed packages as `name @ file:///home/conda/feedstock_root/...` — a path on the conda-forge build machine, useless as a record — and it omits `pip`, `setuptools` and `wheel` unless asked, which would hide a deliberate pin.

## 4. GPU architecture: the check that must not be naive

`torch.cuda.get_arch_list()` returns the GPU architectures a given torch binary was **compiled for**. Testing whether the device's own architecture string appears in that list is the obvious check, and it is wrong.

CUDA guarantees **minor-version binary compatibility**: a cubin built for `sm_XY` runs on any device of the same major `X` whose minor is greater than or equal to `Y`. So a torch whose list stops at `sm_120` runs natively on an `sm_121` device, and an exact-match test reports a healthy install as broken. That is not hypothetical — it is the situation on the reference machine (§5), where the naive check would have condemned an environment that went on to run the full demo.

`conda.sh` therefore parses each tag into `(major, minor)` and reports one of four outcomes: **native** (the exact tag is present), **binary-compatible** (a lower minor of the same major covers it), **PTX JIT only** (a `compute_XX` entry exists, so it works but pays a compile on first launch of every kernel), or **no coverage**, which is the only real error. Then it runs actual GPU work, because the arch list is a clue about the binary while a completed matmul is evidence.

The failure this guards against is real, just rarer than the false alarm: torch can import, report `cuda.is_available() == True`, and still have no usable kernel for the chip.

## 5. Reference machine and status

`#!target` in `deps.dgx-spark.tsv`: NVIDIA DGX Spark, host `spark-e269`. linux-aarch64, GB10 (Blackwell), compute capability 12.1 (`sm_121`), Ubuntu 24.04, driver 595.71.05 reporting CUDA 13.2, toolkit `nvcc` 13.0 at `/usr/local/cuda-13.0`, Anaconda3 at `~/anaconda3`.

Note that `cuda=` in `#!target` is **the maximum runtime the driver supports**, as reported by `nvidia-smi`, because that is what the script compares against. It is not the installed toolkit version. The distinction does not matter for this repository, which compiles nothing, but it will for any manifest that adds a row building a CUDA extension.

### Verified 2026-08-13

A real install ran into environment `mae_st_test`, isolated with `CONDA_PKGS_DIRS`, `PIP_CACHE_DIR` and `TMPDIR` pointed at a scratch directory so nothing landed in the shared caches — which makes a test install fully removable, since conda hardlinks from the package cache when both are on the same filesystem. `verify` exited 0, all eleven required packages import, and the resulting environment is 6.4 GB.

Three risks had been predicted before the first install; all three now have outcomes:

1. **A `cp310` wheel might not exist on the cu128 index — it does.** `torch 2.11.0+cu128` and `torchvision 0.26.0+cu128` install cleanly on linux-aarch64 / cp310.
2. **`sm_121` absent from the arch list — happened, and is benign.** See §4.
3. **`pytorchvideo` failing — happened, exactly as predicted.** It installs without error and then fails at import.

**Acceptance test.** The environment was validated end to end by running the repository's own demo (`visualize_video.py`) against the published checkpoint: 300 frames decoded through OpenCV, reconstructed on the GPU, re-encoded through PyAV as h264, 13.7 s wall clock, checkpoint loading reporting `<All keys matched successfully>`. One command exercises torch and CUDA, `cv2`, PyAV demux and mux, and the model. **This, not the package list, is what "the install worked" means.** Note it does not import `util/kinetics.py`, so it proves the environment and not the training path.

### Four defects found in `conda.sh` by reading it, all fixed

Recorded because each is a shape of bug worth recognising again: a `local` declaration swallowed by a same-line comment, so a function silently relied on the caller's variables having the same names; the `00-env` row's `installer` and `source` columns ignored, so the manifest declared one channel while the script hardcoded another, *and* every non-`python` `00-env` row dropped without a word; the arch check of §4; and no record of what was installed, which is what produced `lock.<tag>.txt` and `logs/`.

## 6. Pinning policy

**Everything is pinned exactly**, read back from the lock. The reasoning is standard practice for a research machine: fix what works, and test migrations exhaustively in a controlled environment rather than discovering them on the machine people depend on.

Pinning torch closes more than it appears to: torch declares its own CUDA stack (`cuda-toolkit`, `cudnn`, `nccl`, `cusparselt`, `nvshmem`) with exact `==`, so those need no rows. Roughly twenty pure-Python transitives still float — the dependencies of tensorboard and matplotlib, and torch's soft ones. The manifest names them explicitly in a `WHAT THIS FILE DOES NOT PIN` section instead of implying full closure.

**Those versions are deliberately not tracked.** Since the lock is discarded once the manifest is pinned, nothing in the repository records which `protobuf` or `grpcio` a given environment had. That is a real limitation and it is stated rather than papered over: a committed lock would *appear* to close the gap while actually being a second, unexplained copy of the version numbers, ageing separately from the manifest. The honest closure is `pip install -c` with a lock supplied as a constraints file at install time — a mechanism decision not yet taken. Until it is, the answer to "which minor version of a transitive was running in August" is: not recorded, and that is on purpose rather than by oversight.

**What pinning caught immediately:** `opencv-python-headless` had been left on `latest` and had crossed a **major** version on its own, to OpenCV 5.x. It was working, but nothing had chosen it.

### Packages whose pins are load-bearing

- **`timm==0.4.5` must not be bumped.** `util/video_vit.py` imports `to_2tuple` from `timm.models.layers` and subclasses `Attention` passing `qk_scale`. timm 0.9 moved `timm.models.layers` to `timm.layers` and removed `qk_scale`. The 0.4.5 wheel was inspected directly: no `torch._six`, no `np.float`, every needed symbol present, and it ships `py3-none-any`, so there is no architecture risk in this ancient pin.
- **`numpy` is pinned to hold a deprecation shim.** `util/video_vit.py` imports a private numpy symbol that numpy 2 still serves only through a compatibility shim. The symbol is never used, so the real fix is a one-line deletion in the repository; until someone makes it, the pin is what keeps the code importable. This is *not* the upstream `numpy==1.23.5` pin, which is unnecessary — the repository contains no `np.float` / `np.int` / `np.bool`, and torch pulls numpy 2 regardless.
- **`setuptools` is held below 70.** setuptools 81 removed `pkg_resources`, which old sdists still import. torch 2.11 independently requires `setuptools<82`, so this sits inside both bounds.
- **`pytorchvideo==0.1.5` is installed and expected to fail.** Its `transforms` package imports `torchvision.transforms.functional_tensor`, removed in torchvision 0.17. `required=no` makes the installer report the failure and continue. The repository imports it unconditionally at module top level, so this failure blocks training until the source is edited — a code fix, out of scope here.

### Deliberate omissions

`detectron2` (built from source by the old venv scripts, imported nowhere), `torchaudio` (imported nowhere), the ActivityNet crawler's `youtube-dl`/`yt-dlp` (the videos are produced locally), the `fvcore`/`yacs`/`scipy`/`pandas` group present in `environment.yml` but unreachable from the training entry point, and a CUDA toolkit inside the environment — which is needed only when something compiles CUDA extensions at install time, and this repository compiles nothing while the torch wheel ships its own runtime.

## 7. Rows that depend on the torch generation

**Established 2026-08-14, while preparing a manifest for a CUDA 11.7 machine.** This section exists because an earlier reading of the manifest was wrong in a way that produces a broken environment while every tool reports success.

That reading split rows two ways: python, torch, torchvision and their index being machine-dependent, everything else a property of the repository and therefore identical everywhere. There is a **third** category — rows that must agree with the *torch generation* that was chosen, and only indirectly with the machine.

- **`numpy`.** Torch wheels are compiled against a specific NumPy ABI, so a pre-NumPy-2 torch cannot run with NumPy 2 installed. Torch 2.0.1 **does not declare numpy among its dependencies at all**, so pip resolves the broken pairing without a word. This is invisible to every resolver, `--resolve-only` included: a constraint nobody declared cannot be checked. Only the functional test in `verify`, or prior knowledge, catches it.
- **`opencv-python-headless`.** Releases from 4.12 onward *require* `numpy>=2`, so on an older torch they drag in the very NumPy that breaks it — from the opposite direction, and equally silently. 4.11.0.86 is the last release that accepts NumPy 1.x, and it ships `cp37-abi3` wheels for both x86_64 and aarch64.
- **The `required` flag on `pytorchvideo`.** It cannot be imported alongside torchvision 0.17 or newer, and is fine before that. On a CUDA 11.7 stack — torchvision 0.15.2 — it should import, which also means the import-time blocker that stops training on newer stacks does not exist there. Whether a failure is *expected* is therefore a property of the manifest, not of the package.

The operating rule this produced, now step 9 of `README.md`: **a pin carried over from another machine's lock is unvouched.** Relax it to a range and let the local lock re-pin it. Only pins with a stated reason that holds everywhere — `timm==0.4.5` — travel between machines unchanged.

It also raises the bar for the overlay design below: an overlay must be able to *relax* an inherited pin, not merely tighten one.

## 8. Still to write

The **base + overlay split** for manifests. Copying a whole manifest per machine duplicates the rows that never vary and guarantees drift. The agreed direction is a shared base plus a per-host overlay that may override any row by package name, add rows, or remove them; rows are matched by normalised package name, duplicates within a single file are an error rather than a last-one-wins, and every resolved row displays which file it came from.
